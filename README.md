# BIM-Gesture-Assistance
A model that helps user translate BIM into english words

# Setup
This project is built using Python 3.12. Using newer versions (like 3.13+) will cause errors with MediaPipe.

1. Install Python 3.12 (If you don't have it)
Download Link: [Python 3.12.7 (Windows x64)](https://www.python.org/downloads/release/python-3120/)

Crucial: During installation, check the box that says "Add Python to PATH".

2. Prepare the Environment
Open your terminal and run:
Navigate to project folder (it should look like "C:\Users\USER\Documents\Project\BIM-Gesture-Assistance>" this)

Create a virtual environment specifically using 3.12:
`py -3.12 -m venv venv`

Activate the environment:
`.\venv\Scripts\activate`

!! caution:If you get a 'scripts disabled' error, run: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process` !!

3. Install Libraries:
`pip install -r requirements.txt`

4. Run the app:
`python App/src/app.py`