import sounddevice as sd
import soundfile as sf

duration = 600  # seconds
samplerate = 16000
filename = "m4.wav"

print(f"🎤 Recording your voice for {duration} seconds...")
sd.default.device = (2, None)  # use device ID 2 for input
recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='float32')
sd.wait()
sf.write(filename, recording, samplerate)
print(f"✅ Reference voice saved as {filename}")
