import os
import re
import sys
import json
import time
import uuid
import random
import secrets
import sqlite3
import httpx
from datetime import datetime, timezone
from functools import wraps

from flask import Flask, request, jsonify, send_from_directory, session, g, Response, redirect
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
import assemblyai as aai
from google import genai
from google.genai import types
from google.genai import errors as genai_errors
from config import genai_key, aai_key

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'frontend'))
DB_PATH = os.path.join(BASE_DIR, 'healthsync.db')
SECRET_PATH = os.path.join(BASE_DIR, '.flask_secret')
UPLOAD_DIR = os.path.join(BASE_DIR, 'uploads')

aai.settings.api_key = aai_key
aai.settings.http_timeout = 30  # seconds per request to AssemblyAI
# Fail after 90s instead of hanging forever if Gemini never answers
client = genai.Client(api_key=genai_key, http_options=types.HttpOptions(timeout=90_000))

sys.stdout.reconfigure(line_buffering=True)  # show log lines immediately, even when not run in a terminal

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200 MB upload limit
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'


def load_secret_key():
    """Keep the session key on disk so patients stay logged in across server restarts."""
    if os.path.exists(SECRET_PATH):
        with open(SECRET_PATH) as f:
            return f.read().strip()
    key = secrets.token_hex(32)
    with open(SECRET_PATH, 'w') as f:
        f.write(key)
    return key


app.secret_key = load_secret_key()

ALLOWED_AUDIO_EXTS = {'.wav', '.mp3', '.m4a', '.ogg', '.webm', '.mp4', '.aac', '.flac', '.mov'}


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

SCHEMA = """
CREATE TABLE IF NOT EXISTS patients (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_code  TEXT UNIQUE NOT NULL,
    name          TEXT NOT NULL,
    email         TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    created_at    TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS checkins (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id    INTEGER NOT NULL REFERENCES patients(id),
    checked_in_at TEXT NOT NULL,
    status        TEXT NOT NULL DEFAULT 'waiting'   -- waiting | seen
);
CREATE TABLE IF NOT EXISTS visits (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id      INTEGER NOT NULL REFERENCES patients(id),
    status          TEXT NOT NULL DEFAULT 'draft',  -- draft | approved
    transcript      TEXT,                           -- already PII-redacted
    note            TEXT,
    patient_summary TEXT,
    patient_goals   TEXT,                           -- JSON list
    diet_plan       TEXT,                           -- JSON list
    reminders       TEXT,                           -- JSON list of {title, time, repeat}
    created_at      TEXT NOT NULL,
    approved_at     TEXT
);
-- Cached AI translations / simplifications of a visit's patient-facing content
CREATE TABLE IF NOT EXISTS visit_translations (
    visit_id  INTEGER NOT NULL REFERENCES visits(id) ON DELETE CASCADE,
    lang      TEXT NOT NULL,
    level     TEXT NOT NULL,
    content   TEXT NOT NULL,   -- JSON {patient_summary, patient_goals, diet_plan, reminder_titles}
    PRIMARY KEY (visit_id, lang, level)
);
-- Cached AI translations of the app's interface text
CREATE TABLE IF NOT EXISTS ui_translations (
    lang        TEXT NOT NULL,
    source_hash TEXT NOT NULL,
    strings     TEXT NOT NULL,
    PRIMARY KEY (lang, source_hash)
);
"""

# Patient settings and their allowed values (first = default)
LANGUAGES = {
    'en': 'English', 'es': 'Español', 'hi': 'हिन्दी', 'zh': '中文', 'fr': 'Français', 'pt': 'Português',
    'vi': 'Tiếng Việt', 'tl': 'Tagalog', 'ko': '한국어', 'ru': 'Русский',
}
LANGUAGE_NAMES_EN = {
    'en': 'English', 'es': 'Spanish', 'hi': 'Hindi', 'zh': 'Simplified Chinese', 'fr': 'French', 'pt': 'Portuguese',
    'vi': 'Vietnamese', 'tl': 'Tagalog', 'ko': 'Korean', 'ru': 'Russian',
}
SETTING_CHOICES = {
    'language': list(LANGUAGES),
    'reading_level': ['standard', 'simple'],
    'text_size': ['normal', 'large', 'xlarge'],
}
DEFAULT_SETTINGS = {k: v[0] for k, v in SETTING_CHOICES.items()}


def get_db():
    if 'db' not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
        g.db.execute('PRAGMA foreign_keys = ON')
    return g.db


@app.teardown_appcontext
def close_db(_exc):
    db = g.pop('db', None)
    if db is not None:
        db.close()


def init_db():
    with sqlite3.connect(DB_PATH) as db:
        db.executescript(SCHEMA)
        # Migration: add settings column to databases created before it existed
        cols = {r[1] for r in db.execute('PRAGMA table_info(patients)')}
        if 'settings' not in cols:
            db.execute("ALTER TABLE patients ADD COLUMN settings TEXT NOT NULL DEFAULT '{}'")


def patient_settings(row):
    try:
        saved = json.loads(row['settings'] or '{}')
    except (ValueError, IndexError, KeyError):
        saved = {}
    return {k: (saved.get(k) if saved.get(k) in SETTING_CHOICES[k] else DEFAULT_SETTINGS[k]) for k in SETTING_CHOICES}


def parse_json_object(text):
    """Parse a model's JSON answer. Models occasionally wrap the object in a list or code fences."""
    text = text.strip()
    if text.startswith('```'):
        text = text.strip('`').split('\n', 1)[1] if '\n' in text else text.strip('`')
    data = json.loads(text)
    if isinstance(data, list) and len(data) == 1 and isinstance(data[0], dict):
        data = data[0]
    if not isinstance(data, dict):
        raise ValueError(f"expected a JSON object, got {type(data).__name__}")
    return data


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def new_patient_code(db):
    while True:
        code = f"HS-{random.randint(1000, 9999)}"
        if not db.execute('SELECT 1 FROM patients WHERE patient_code = ?', (code,)).fetchone():
            return code


def visit_to_dict(row, include_clinical=True):
    d = {
        "id": row["id"],
        "patient_id": row["patient_id"],
        "status": row["status"],
        "patient_summary": row["patient_summary"],
        "patient_goals": json.loads(row["patient_goals"] or '[]'),
        "diet_plan": json.loads(row["diet_plan"] or '[]'),
        "reminders": json.loads(row["reminders"] or '[]'),
        "created_at": row["created_at"],
        "approved_at": row["approved_at"],
    }
    if include_clinical:
        d["note"] = row["note"]
        d["transcript"] = row["transcript"]
    return d


def patient_public(row):
    return {"id": row["id"], "patient_code": row["patient_code"], "name": row["name"], "email": row["email"],
            "settings": patient_settings(row), "languages": LANGUAGES}


# ---------------------------------------------------------------------------
# AI pipeline: AssemblyAI (transcribe + redact) -> Gemini (structure)
# ---------------------------------------------------------------------------

# Identity details are removed before any AI model sees the text.
# Medical details (drugs, conditions, procedures, ages, times) are kept because the note needs them.
PII_POLICIES = [
    aai.PIIRedactionPolicy.person_name,
    aai.PIIRedactionPolicy.phone_number,
    aai.PIIRedactionPolicy.email_address,
    aai.PIIRedactionPolicy.date_of_birth,
    aai.PIIRedactionPolicy.location_address,
    aai.PIIRedactionPolicy.location_address_street,
    aai.PIIRedactionPolicy.location_zip,
    aai.PIIRedactionPolicy.location_coordinate,
    aai.PIIRedactionPolicy.healthcare_number,
    aai.PIIRedactionPolicy.us_social_security_number,
    aai.PIIRedactionPolicy.drivers_license,
    aai.PIIRedactionPolicy.passport_number,
    aai.PIIRedactionPolicy.account_number,
    aai.PIIRedactionPolicy.banking_information,
    aai.PIIRedactionPolicy.credit_card_number,
    aai.PIIRedactionPolicy.username,
    aai.PIIRedactionPolicy.password,
    aai.PIIRedactionPolicy.ip_address,
    aai.PIIRedactionPolicy.url,
]

TRANSCRIPTION_CONFIG = aai.TranscriptionConfig(
    speaker_labels=True,
    redact_pii=True,
    redact_pii_policies=PII_POLICIES,
    redact_pii_sub=aai.PIISubstitutionPolicy.entity_name,  # "John" -> "[PERSON_NAME]"
)


def transcribe_with_retry(audio_path, attempts=3):
    """Transcribe + redact with AssemblyAI, retrying temporary failures."""
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            transcript = aai.Transcriber(config=TRANSCRIPTION_CONFIG).transcribe(audio_path)
            if transcript.status != aai.TranscriptStatus.error:
                if transcript.utterances:
                    return "\n".join(f"Speaker {u.speaker}: {u.text}" for u in transcript.utterances)
                return transcript.text
            last_error = transcript.error
            if any(w in str(last_error).lower() for w in ("auth", "api key", "no spoken audio")):
                break
        except Exception as e:  # network errors, timeouts, server errors
            last_error = e
        if attempt < attempts:
            delay = 2 ** attempt + random.uniform(0, 1)
            print(f"[assemblyai] failed ({last_error}), retry {attempt}/{attempts - 1} in {delay:.1f}s")
            time.sleep(delay)
    raise RuntimeError(f"AssemblyAI transcription failed: {last_error}")


# Gemini sometimes returns 503 "model is overloaded" or 429 "rate limited".
# Tried in order: if one is overloaded or out of quota, we fall back to the next.
# Note-writing uses the strongest models; translation is simpler, so it uses fast "lite" models first.
GEMINI_MODELS = ['gemini-3.8-flash', 'gemini-3.6-flash', 'gemini-3.5-flash-lite']
FAST_MODELS = ['gemini-3.5-flash-lite', 'gemini-3.1-flash-lite', 'gemini-3.6-flash']
RETRYABLE_CODES = {429, 500, 503, 504}
MAX_ATTEMPTS_PER_MODEL = 2  # with backups available, move on quickly rather than waiting
# Max seconds to wait on one model before trying the next (a hung model shouldn't stall the app)
NOTE_TIMEOUT_S = 60
FAST_TIMEOUT_S = 20


def generate_with_retry(prompt, config, models=None, timeout_s=None):
    models = models or GEMINI_MODELS
    if timeout_s is None:
        timeout_s = FAST_TIMEOUT_S if models is FAST_MODELS else NOTE_TIMEOUT_S
    config = config.model_copy(update={"http_options": types.HttpOptions(timeout=timeout_s * 1000)})
    last_error = None
    for model in models:
        for attempt in range(1, MAX_ATTEMPTS_PER_MODEL + 1):
            t0 = time.time()
            try:
                response = client.models.generate_content(model=model, contents=prompt, config=config)
                print(f"[gemini] answered by {model} in {time.time() - t0:.1f}s")
                return response
            except httpx.TimeoutException as e:
                last_error = e
                print(f"[gemini] {model} timed out after {time.time() - t0:.0f}s, moving on")
                break  # a hung model will likely hang again; try the next one
            except genai_errors.APIError as e:
                last_error = e
                if e.code not in RETRYABLE_CODES:
                    raise  # bad key, bad request, etc. -- retrying won't help
                if e.code == 429 and 'RESOURCE_EXHAUSTED' in str(e.status or e):
                    print(f"[gemini] {model} is out of quota, skipping to the next model")
                    break  # quota won't come back in a few seconds
                if e.code == 504:
                    print(f"[gemini] {model} took too long (504) after {time.time() - t0:.0f}s, moving on")
                    break  # it already made us wait; don't wait on it twice
                if attempt == MAX_ATTEMPTS_PER_MODEL:
                    print(f"[gemini] {model} still failing after {attempt} tries, moving on")
                    break
                delay = min(2 ** attempt, 20) + random.uniform(0, 1)
                print(f"[gemini] {model} returned {e.code}, retry {attempt}/{MAX_ATTEMPTS_PER_MODEL - 1} in {delay:.1f}s")
                time.sleep(delay)
    raise RuntimeError("The AI service is busy right now. Please try again in a minute.") from last_error


JSON_CONFIG = types.GenerateContentConfig(response_mime_type="application/json")


def generate_json(prompt, models=None):
    """Ask Gemini for a JSON object. If the answer is malformed JSON, ask once more starting from the next model."""
    models = models or GEMINI_MODELS
    timeout_s = FAST_TIMEOUT_S if models is FAST_MODELS else NOTE_TIMEOUT_S
    try:
        return parse_json_object(generate_with_retry(prompt, JSON_CONFIG, models, timeout_s).text)
    except ValueError as e:  # json.JSONDecodeError is a ValueError
        print(f"[gemini] malformed JSON ({e}); asking again with the next model")
        rotated = models[1:] + models[:1]
        return parse_json_object(generate_with_retry(prompt, JSON_CONFIG, rotated, timeout_s).text)


SCRIBE_PROMPT = """You are a medical scribe. Below is a PII-redacted transcript of a doctor-patient visit.
Identity details were replaced with tags like [PERSON_NAME]; never guess or reconstruct them.

Rules:
- Use only information stated in the transcript. Do not invent symptoms, vitals, diagnoses, or instructions.
  If something was not discussed, leave it out (or write "Not discussed" in the clinical note).
- Refer to the person as "the patient" in the clinical note.
- Do not make assumptions based on age, gender, race, or background beyond what was said.
- The patient summary is written TO the patient ("you"), in plain language a 12-year-old could follow.

Transcript:
\"\"\"
{transcript}
\"\"\"

Return a JSON object with exactly these keys:
{{
  "note": "Clinical note in SOAP format with headings Subjective, Objective, Assessment, Plan, separated by line breaks.",
  "patient_summary": "2-4 friendly sentences explaining what was found and what to do next.",
  "patient_goals": ["Short, actionable goal", "..."],
  "diet_plan": ["Specific diet instruction", "..."],
  "reminders": [
    {{"title": "What to do, e.g. 'Take lisinopril 10 mg'", "time": "HH:MM in 24h, best guess from the visit (e.g. 08:00 for morning)", "repeat": "daily | weekly | once"}}
  ]
}}"""


def normalize_ai_output(data):
    def str_list(v):
        return [str(x).strip() for x in (v or []) if str(x).strip()]

    reminders = []
    for r in data.get('reminders') or []:
        if isinstance(r, str):
            r = {"title": r}
        title = str(r.get('title', '')).strip()
        if not title:
            continue
        t = str(r.get('time', '')).strip()
        if not re.fullmatch(r'([01]\d|2[0-3]):[0-5]\d', t):
            t = '09:00'
        repeat = str(r.get('repeat', 'daily')).lower()
        if repeat not in ('daily', 'weekly', 'once'):
            repeat = 'daily'
        reminders.append({"title": title, "time": t, "repeat": repeat})

    return {
        "note": str(data.get('note') or '').strip(),
        "patient_summary": str(data.get('patient_summary') or '').strip(),
        "patient_goals": str_list(data.get('patient_goals')),
        "diet_plan": str_list(data.get('diet_plan')),
        "reminders": reminders,
    }


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

@app.route('/')
def index():
    return redirect('/app')


@app.route('/doctor')
def serve_doctor():
    return send_from_directory(FRONTEND_DIR, 'doctor.html')


@app.route('/app')
@app.route('/patient')
def serve_patient():
    return send_from_directory(FRONTEND_DIR, 'patient.html')


# ---------------------------------------------------------------------------
# Patient API (requires login, except signup/login)
# ---------------------------------------------------------------------------

def patient_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        pid = session.get('patient_id')
        if not pid:
            return jsonify({"error": "Please log in."}), 401
        row = get_db().execute('SELECT * FROM patients WHERE id = ?', (pid,)).fetchone()
        if not row:
            session.clear()
            return jsonify({"error": "Please log in."}), 401
        g.patient = row
        return fn(*args, **kwargs)
    return wrapper


@app.route('/api/signup', methods=['POST'])
def signup():
    body = request.get_json(silent=True) or {}
    name = (body.get('name') or '').strip()
    email = (body.get('email') or '').strip().lower()
    password = body.get('password') or ''
    if not name or not email or not password:
        return jsonify({"error": "Name, email and password are all required."}), 400
    if not re.fullmatch(r'[^@\s]+@[^@\s]+\.[^@\s]+', email):
        return jsonify({"error": "Please enter a valid email address."}), 400
    if len(password) < 6:
        return jsonify({"error": "Password must be at least 6 characters."}), 400

    db = get_db()
    if db.execute('SELECT 1 FROM patients WHERE email = ?', (email,)).fetchone():
        return jsonify({"error": "An account with that email already exists. Try logging in."}), 409
    code = new_patient_code(db)
    cur = db.execute(
        'INSERT INTO patients (patient_code, name, email, password_hash, created_at) VALUES (?, ?, ?, ?, ?)',
        (code, name, email, generate_password_hash(password), now_iso()),
    )
    db.commit()
    session.clear()
    session['patient_id'] = cur.lastrowid
    session.permanent = True
    row = db.execute('SELECT * FROM patients WHERE id = ?', (cur.lastrowid,)).fetchone()
    data = patient_public(row)
    data["checked_in"] = False
    return jsonify(data), 201


@app.route('/api/login', methods=['POST'])
def login():
    body = request.get_json(silent=True) or {}
    email = (body.get('email') or '').strip().lower()
    password = body.get('password') or ''
    row = get_db().execute('SELECT * FROM patients WHERE email = ?', (email,)).fetchone()
    if not row or not check_password_hash(row['password_hash'], password):
        return jsonify({"error": "Incorrect email or password."}), 401
    session.clear()
    session['patient_id'] = row['id']
    session.permanent = True
    data = patient_public(row)
    data["checked_in"] = bool(get_db().execute(
        "SELECT 1 FROM checkins WHERE patient_id = ? AND status = 'waiting'", (row['id'],)
    ).fetchone())
    return jsonify(data)


@app.route('/api/logout', methods=['POST'])
def logout():
    session.clear()
    return jsonify({"ok": True})


@app.route('/api/me')
@patient_required
def me():
    db = get_db()
    checkin = db.execute(
        "SELECT * FROM checkins WHERE patient_id = ? AND status = 'waiting' ORDER BY id DESC LIMIT 1",
        (g.patient['id'],),
    ).fetchone()
    data = patient_public(g.patient)
    data["checked_in"] = bool(checkin)
    return jsonify(data)


@app.route('/api/checkin', methods=['POST'])
@patient_required
def checkin():
    db = get_db()
    existing = db.execute(
        "SELECT 1 FROM checkins WHERE patient_id = ? AND status = 'waiting'", (g.patient['id'],)
    ).fetchone()
    if not existing:
        db.execute('INSERT INTO checkins (patient_id, checked_in_at) VALUES (?, ?)', (g.patient['id'], now_iso()))
        db.commit()
    return jsonify({"checked_in": True})


LOCALIZE_PROMPT = """You are helping a patient understand their doctor-approved care plan.
Rewrite the JSON below {task}.

Strict rules (patient safety):
- Keep the meaning exactly the same. Do not add, remove, or change any medical advice.
- Keep medication names, doses, numbers, and units exactly as written (e.g. "lisinopril 10 mg" stays "lisinopril 10 mg").
- Keep the same number of items in every list, in the same order.
- Write warmly and clearly, speaking directly to the patient as "you".

Input JSON:
{payload}

Return JSON with exactly the same keys and list lengths."""


def localize_visit(db, visit, settings):
    """Translate / simplify the patient-facing parts of a visit, cached per (visit, language, level)."""
    lang, level = settings['language'], settings['reading_level']
    if lang == 'en' and level == 'standard':
        return visit

    cached = db.execute('SELECT content FROM visit_translations WHERE visit_id = ? AND lang = ? AND level = ?',
                        (visit['id'], lang, level)).fetchone()
    if cached:
        content = json.loads(cached['content'])
    else:
        source = {
            "patient_summary": visit['patient_summary'],
            "patient_goals": visit['patient_goals'],
            "diet_plan": visit['diet_plan'],
            "reminder_titles": [r['title'] for r in visit['reminders']],
        }
        parts = []
        if level == 'simple':
            parts.append('in very simple, everyday words a 10-year-old could understand (short sentences, no medical jargon)')
        if lang != 'en':
            parts.append(f'translated into {LANGUAGE_NAMES_EN[lang]}')
        prompt = LOCALIZE_PROMPT.format(task=' and '.join(parts), payload=json.dumps(source, ensure_ascii=False, indent=2))
        try:
            content = generate_json(prompt, FAST_MODELS)
            # Safety net: if the model changed list lengths, don't risk a mismatched plan
            for k in ('patient_goals', 'diet_plan', 'reminder_titles'):
                if not isinstance(content.get(k), list) or len(content[k]) != len(source[k]):
                    raise ValueError(f"translation changed the number of items in {k}")
        except Exception as e:
            print(f"[localize] visit {visit['id']} -> {lang}/{level} failed: {e!r}; showing original")
            return {**visit, "localized": False}
        db.execute('INSERT OR REPLACE INTO visit_translations (visit_id, lang, level, content) VALUES (?, ?, ?, ?)',
                   (visit['id'], lang, level, json.dumps(content, ensure_ascii=False)))
        db.commit()

    out = dict(visit)
    out['patient_summary'] = content.get('patient_summary') or visit['patient_summary']
    out['patient_goals'] = content['patient_goals']
    out['diet_plan'] = content['diet_plan']
    out['reminders'] = [{**r, 'title': t} for r, t in zip(visit['reminders'], content['reminder_titles'])]
    out['localized'] = True
    return out


@app.route('/api/my-visits')
@patient_required
def my_visits():
    db = get_db()
    rows = db.execute(
        "SELECT * FROM visits WHERE patient_id = ? AND status = 'approved' ORDER BY approved_at DESC",
        (g.patient['id'],),
    ).fetchall()
    settings = patient_settings(g.patient)
    # Patients see their plan and summary (in their language/reading level); the clinical note stays as the doctor wrote it
    return jsonify([localize_visit(db, visit_to_dict(r, include_clinical=True), settings) for r in rows])


@app.route('/api/settings', methods=['PATCH'])
@patient_required
def update_settings():
    body = request.get_json(silent=True) or {}
    settings = patient_settings(g.patient)
    for k, v in body.items():
        if k in SETTING_CHOICES:
            if v not in SETTING_CHOICES[k]:
                return jsonify({"error": f"Invalid value for {k}."}), 400
            settings[k] = v
    db = get_db()
    db.execute('UPDATE patients SET settings = ? WHERE id = ?', (json.dumps(settings), g.patient['id']))
    db.commit()
    return jsonify(settings)


@app.route('/api/account', methods=['PATCH'])
@patient_required
def update_account():
    body = request.get_json(silent=True) or {}
    db = get_db()
    if 'name' in body:
        name = (body.get('name') or '').strip()
        if not name:
            return jsonify({"error": "Name can't be empty."}), 400
        db.execute('UPDATE patients SET name = ? WHERE id = ?', (name, g.patient['id']))
    if 'new_password' in body:
        if not check_password_hash(g.patient['password_hash'], body.get('current_password') or ''):
            return jsonify({"error": "Your current password is incorrect."}), 400
        if len(body['new_password'] or '') < 6:
            return jsonify({"error": "New password must be at least 6 characters."}), 400
        db.execute('UPDATE patients SET password_hash = ? WHERE id = ?',
                   (generate_password_hash(body['new_password']), g.patient['id']))
    db.commit()
    row = db.execute('SELECT * FROM patients WHERE id = ?', (g.patient['id'],)).fetchone()
    return jsonify(patient_public(row))


@app.route('/api/account', methods=['DELETE'])
@patient_required
def delete_account():
    """Privacy: a patient can permanently delete their account and every visit record."""
    body = request.get_json(silent=True) or {}
    if not check_password_hash(g.patient['password_hash'], body.get('password') or ''):
        return jsonify({"error": "Password is incorrect."}), 400
    db = get_db()
    pid = g.patient['id']
    db.execute('DELETE FROM visit_translations WHERE visit_id IN (SELECT id FROM visits WHERE patient_id = ?)', (pid,))
    db.execute('DELETE FROM visits WHERE patient_id = ?', (pid,))
    db.execute('DELETE FROM checkins WHERE patient_id = ?', (pid,))
    db.execute('DELETE FROM patients WHERE id = ?', (pid,))
    db.commit()
    session.clear()
    return jsonify({"ok": True})


@app.route('/api/ui-strings', methods=['POST'])
@patient_required
def ui_strings():
    """Translate the app's interface text (sent by the page in English) once per language, then cache it."""
    import hashlib
    body = request.get_json(silent=True) or {}
    lang = body.get('lang')
    strings = body.get('strings')
    if lang not in LANGUAGES or not isinstance(strings, dict):
        return jsonify({"error": "Bad request."}), 400
    if lang == 'en':
        return jsonify(strings)
    source_hash = hashlib.sha256(json.dumps(strings, sort_keys=True).encode()).hexdigest()[:16]
    db = get_db()
    cached = db.execute('SELECT strings FROM ui_translations WHERE lang = ? AND source_hash = ?',
                        (lang, source_hash)).fetchone()
    if cached:
        return jsonify(json.loads(cached['strings']))
    prompt = (f"Translate the values of this JSON object from English into {LANGUAGE_NAMES_EN[lang]} for a patient-facing "
              "health app. Keep every key unchanged. Keep placeholders like {name} or {code} exactly as they are. "
              "Keep emoji. Keep it short and natural, like a real app's interface.\n\n"
              + json.dumps(strings, ensure_ascii=False, indent=2))
    try:
        translated = generate_json(prompt, FAST_MODELS)
        translated = {k: str(translated.get(k) or v) for k, v in strings.items()}  # fall back per missing key
    except Exception as e:
        print(f"[ui-strings] {lang} failed: {e!r}")
        return jsonify({"error": "Translation is unavailable right now."}), 503
    db.execute('INSERT OR REPLACE INTO ui_translations (lang, source_hash, strings) VALUES (?, ?, ?)',
               (lang, source_hash, json.dumps(translated, ensure_ascii=False)))
    db.commit()
    return jsonify(translated)


def ics_escape(text):
    return text.replace('\\', '\\\\').replace(';', '\\;').replace(',', '\\,').replace('\n', '\\n')


@app.route('/api/visits/<int:visit_id>/reminders.ics')
@patient_required
def reminders_ics(visit_id):
    row = get_db().execute(
        "SELECT * FROM visits WHERE id = ? AND patient_id = ? AND status = 'approved'",
        (visit_id, g.patient['id']),
    ).fetchone()
    if not row:
        return jsonify({"error": "Visit not found."}), 404

    reminders = json.loads(row['reminders'] or '[]')
    only = request.args.get('i')
    if only is not None and only.isdigit() and int(only) < len(reminders):
        reminders = [reminders[int(only)]]

    today = datetime.now().strftime('%Y%m%d')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    lines = ['BEGIN:VCALENDAR', 'VERSION:2.0', 'PRODID:-//HealthSync//Reminders//EN', 'CALSCALE:GREGORIAN']
    for r in reminders:
        hh, mm = r['time'].split(':')
        lines += [
            'BEGIN:VEVENT',
            f'UID:{uuid.uuid4()}@healthsync',
            f'DTSTAMP:{stamp}',
            f'DTSTART:{today}T{hh}{mm}00',
            'DURATION:PT10M',
            f'SUMMARY:{ics_escape(r["title"])}',
            'DESCRIPTION:Reminder from your HealthSync care plan',
        ]
        if r['repeat'] == 'daily':
            lines.append('RRULE:FREQ=DAILY')
        elif r['repeat'] == 'weekly':
            lines.append('RRULE:FREQ=WEEKLY')
        lines += ['BEGIN:VALARM', 'ACTION:DISPLAY', f'DESCRIPTION:{ics_escape(r["title"])}',
                  'TRIGGER:PT0M', 'END:VALARM', 'END:VEVENT']
    lines.append('END:VCALENDAR')
    return Response('\r\n'.join(lines) + '\r\n', mimetype='text/calendar',
                    headers={'Content-Disposition': f'attachment; filename="healthsync-reminders-{visit_id}.ics"'})


# ---------------------------------------------------------------------------
# Doctor API (no login for the hackathon demo: single doctor)
# ---------------------------------------------------------------------------

@app.route('/api/doctor/waiting')
def doctor_waiting():
    rows = get_db().execute(
        """SELECT p.id, p.name, p.patient_code, c.checked_in_at
           FROM checkins c JOIN patients p ON p.id = c.patient_id
           WHERE c.status = 'waiting' ORDER BY c.checked_in_at""",
    ).fetchall()
    return jsonify([dict(r) for r in rows])


@app.route('/api/doctor/patients')
def doctor_search_patients():
    q = (request.args.get('q') or '').strip()
    if len(q) < 1:
        return jsonify([])
    like = f'%{q}%'
    rows = get_db().execute(
        'SELECT id, name, patient_code FROM patients WHERE name LIKE ? OR patient_code LIKE ? ORDER BY name LIMIT 10',
        (like, like),
    ).fetchall()
    return jsonify([dict(r) for r in rows])


@app.route('/api/doctor/visits', methods=['POST'])
def doctor_create_visit():
    """Audio in -> redacted transcript -> AI draft. Nothing reaches the patient until approved."""
    patient_id = request.form.get('patient_id', type=int)
    db = get_db()
    patient = db.execute('SELECT * FROM patients WHERE id = ?', (patient_id,)).fetchone() if patient_id else None
    if not patient:
        return jsonify({"error": "Choose a patient before recording."}), 400
    if 'audio' not in request.files:
        return jsonify({"error": "Missing audio file."}), 400

    audio_file = request.files['audio']
    ext = os.path.splitext(audio_file.filename or '')[1].lower()
    if ext and ext not in ALLOWED_AUDIO_EXTS:
        return jsonify({"error": f"Unsupported file type '{ext}'. Upload an audio file like .mp3, .m4a or .wav."}), 400

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    audio_path = os.path.join(UPLOAD_DIR, f'{uuid.uuid4().hex}{ext or ".webm"}')
    audio_file.save(audio_path)
    try:
        size = os.path.getsize(audio_path)
        print(f"[audio] received {audio_file.filename} ({size / 1024:.1f} KB) for patient #{patient_id}")
        if size < 8_000:
            return jsonify({"error": "The recording is almost empty. Check that a microphone is connected and record for at least a few seconds, or upload a recording instead."}), 400

        t0 = time.time()
        try:
            transcript = transcribe_with_retry(audio_path)
        except Exception as e:
            print(f"[assemblyai] giving up: {e}")
            return jsonify({"error": f"[Transcription / AssemblyAI] {e}"}), 500
    finally:
        # Privacy: the raw recording is never kept once transcription is done
        if os.path.exists(audio_path):
            os.remove(audio_path)

    t1 = time.time()
    print(f"[timing] transcription took {t1 - t0:.1f}s ({len(transcript or '')} chars)")
    if not transcript or len(transcript.strip()) < 20:
        return jsonify({"error": "No speech was detected in this recording. Check the microphone, or upload a different file."}), 400

    try:
        result = normalize_ai_output(generate_json(SCRIBE_PROMPT.format(transcript=transcript)))
    except Exception as e:
        print(f"[gemini] giving up: {e!r}")
        return jsonify({"error": f"[Note generation / Gemini] {e}"}), 500
    t2 = time.time()
    print(f"[timing] note generation took {t2 - t1:.1f}s | total {t2 - t0:.1f}s")

    cur = db.execute(
        """INSERT INTO visits (patient_id, status, transcript, note, patient_summary, patient_goals, diet_plan, reminders, created_at)
           VALUES (?, 'draft', ?, ?, ?, ?, ?, ?, ?)""",
        (patient_id, transcript, result['note'], result['patient_summary'], json.dumps(result['patient_goals']),
         json.dumps(result['diet_plan']), json.dumps(result['reminders']), now_iso()),
    )
    db.commit()
    row = db.execute('SELECT * FROM visits WHERE id = ?', (cur.lastrowid,)).fetchone()
    return jsonify(visit_to_dict(row)), 201


@app.route('/api/doctor/visits/<int:visit_id>/approve', methods=['POST'])
def doctor_approve_visit(visit_id):
    """Doctor approves (optionally with edits). Only now does the patient see it."""
    db = get_db()
    row = db.execute('SELECT * FROM visits WHERE id = ?', (visit_id,)).fetchone()
    if not row:
        return jsonify({"error": "Visit not found."}), 404

    edits = request.get_json(silent=True) or {}
    current = visit_to_dict(row)
    merged = normalize_ai_output({k: edits.get(k, current[k]) for k in
                                  ('note', 'patient_summary', 'patient_goals', 'diet_plan', 'reminders')})

    db.execute(
        """UPDATE visits SET status = 'approved', approved_at = ?, note = ?, patient_summary = ?,
           patient_goals = ?, diet_plan = ?, reminders = ? WHERE id = ?""",
        (now_iso(), merged['note'], merged['patient_summary'], json.dumps(merged['patient_goals']),
         json.dumps(merged['diet_plan']), json.dumps(merged['reminders']), visit_id),
    )
    # The patient has been seen: take them off the waiting list
    db.execute("UPDATE checkins SET status = 'seen' WHERE patient_id = ? AND status = 'waiting'", (row['patient_id'],))
    db.commit()
    patient = db.execute('SELECT name FROM patients WHERE id = ?', (row['patient_id'],)).fetchone()
    return jsonify({"ok": True, "patient_name": patient['name']})


@app.errorhandler(413)
def too_large(_e):
    return jsonify({"error": "That file is too large (max 200 MB)."}), 413


init_db()

if __name__ == '__main__':
    # --https: serve over HTTPS with a temporary self-signed certificate.
    # Needed for other devices (laptop/phone) on your network: browsers only allow
    # the microphone on https:// pages or localhost.
    use_https = '--https' in sys.argv
    print("Starting HealthSync local web server...")
    print("  Doctor portal: /doctor    Patient app: /app")
    if use_https:
        print("HTTPS on: open https://<this-computer's-IP>:8080/... and accept the certificate warning")
    app.run(host='0.0.0.0', port=8080, debug=True, ssl_context='adhoc' if use_https else None)
