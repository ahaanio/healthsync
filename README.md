A lot of the time, during doctor appointments, the patient doesn't remember the instructions that the doctor told them, and they don't
understand what the doctor is saying due to language barriers. Also, the doctors burn out writing notes for every patient. 
HealthSync is an AI scribe that records appointments, creates a summary, and sends that information to the patient. Here is how it
works: 

Patient App - The patient goes to the HealthSync App and logs in to an existing account or creates one. Once they log in, they will
see a patient portal with a home screen, settings, doctor's notes, and profile. When they log in, there will be no information yet
because they haven't had a visit yet. On the home screen, there is also a button that allows you to automatically check in to the
doctor's portal if you are already at the clinic. The patient can also select what language they want, so there will be no language
barrier between the patient and doctor. 

Doctor Portal - In the doctor's portal for HealthSync, the doctor can view patients and see who is in the waiting room with the check
in feature. When it's time for the appointment, all the doctor has to do is select the patient, and hit record. After the appointment is
over, all the doctor has to do is stop the recording and let the agent summarize the appointment into notes. The doctor also has the option to upload an appointment recording and still get the same result. 
These notes are only drafted by the AI, and the doctor will be able to look over them and edit them before approving them and sending them to the patient app. Now, when the patient goes 
to the HealthSync app, they will see that their reminders and doctor's notes tab have been updated. They will then be able to enable 
notifications so they get alerts when to do their tasks, like drinking water or following a specific diet plan. 

Privacy and Safety - To ensure the privacy and safety of the patients, all personal information such as names and addresses are all redacted in the transcript, so the AI doesn't see any of it. 
Also, the AI does not directly send the information, so the doctor can review the notes it created, assuring patients that the information they are getting is credible. 

Built with - Python, HTML, Flask, SQLite, Tailwind CSS, AssemblyAI and Google Gemini. All libraries are in requirements.txt.

How to run: 
Things you need - Python 3, a free API key from Google AI Studio, and a free API key from AssemblyAI 

Run this in the terminal to clone the code files, all you need to do is download the libraries onto your own venv and install the libraries from requirements.txt

```
git clone https://github.com/ahaanio/healthsync.git
cd healthsync
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Next, make a copy of backend/config.example.py named backend/config.py (run `cp backend/config.example.py backend/config.py`), then open config.py and paste your real API keys in place of the placeholder text.

Once your keys are in place, go in your terminal and run `python backend/app.py --https` to start the server. 

Then, go to any browser and search these urls:
For patient - https://localhost:8080/app 

For doctor - https://localhost:8080/doctor
Your browser will say that the connection isn't private, but that's only because the app uses a temporary self-signed certificate. Go ahead and click Advanced and Proceed.
If you want to use the app from another device on your wifi, you can replace localhost with the ip address of the computer the server is running on. Ex: https://YOUR_IP:8080/doctor
To stop the server, press Ctrl+C.
