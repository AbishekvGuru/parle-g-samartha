import subprocess
import time
import os

TEXT_FILE = "transcriptions.txt"

def speak_text(text):
    command = f'powershell -Command "Add-Type –AssemblyName System.Speech; ' \
              f'(New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak(\'{text}\');"'
    subprocess.call(command, shell=True)

last_line = ""  # Track last spoken line

while True:
    if os.path.exists(TEXT_FILE):
        with open(TEXT_FILE, "r") as f:
            lines = [line.strip() for line in f if line.strip()]
            if lines:
                current_line = lines[-1]  # always take the last line
                if current_line != last_line:
                    print("Model Output:", current_line)
                    speak_text(current_line)
                    last_line = current_line
    time.sleep(1)
