import sounddevice as sd
import queue, json, numpy as np, librosa
from resemblyzer import VoiceEncoder
from vosk import Model, KaldiRecognizer
from numpy.linalg import norm

# ---------------- CONFIG ----------------
samplerate = 16000
chunk_duration = 0.5  # seconds -> lower latency
chunk_size = int(samplerate * chunk_duration)
threshold = 0.95
verify_interval = 2.0  # seconds between voice rechecks
frames_since_verify = 0

# ---------------- LOAD MODELS ----------------
encoder = VoiceEncoder()
vosk_model = Model(r"E:\samartha\vosk-model-en-us-0.42-gigaspeech")
recognizer = KaldiRecognizer(vosk_model, samplerate)

# ---------------- REFERENCE VOICE ----------------
ref_audio, _ = librosa.load("m4.wav", sr=samplerate)
ref_embedding = encoder.embed_utterance(ref_audio)

# ---------------- STREAM SETUP ----------------
audio_q = queue.Queue()
sd.default.device = (2, None)

def callback(indata, frames, time, status):
    if status:
        print(status)
    audio_q.put(bytes(indata))

stream = sd.RawInputStream(samplerate=samplerate, blocksize=chunk_size, dtype='int16',
                           channels=1, callback=callback)

print("🎤 Listening with low latency... (Ctrl+C to stop)")
with stream:
    try:
        while True:
            data = audio_q.get()
            wav = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            frames_since_verify += chunk_duration

            # Verify every few seconds, not every chunk
            if frames_since_verify >= verify_interval:
                embedding = encoder.embed_utterance(wav)
                sim = np.dot(ref_embedding, embedding) / (norm(ref_embedding)*norm(embedding))
                frames_since_verify = 0
            else:
                sim = 1.0  # assume continuous speaking

            if sim > threshold:
                if recognizer.AcceptWaveform(data):
                    result = json.loads(recognizer.Result())
                    text = result.get("text", "")
                    if text:
                        print(f"🗣️ {text}")
                else:
                    partial = json.loads(recognizer.PartialResult())
                    part = partial.get("partial", "")
                    if part:
                        print(f"🕓 {part}", end="\r")
            else:
                print( end="\r")
    except KeyboardInterrupt:
        print("\n🛑 Stopped")
