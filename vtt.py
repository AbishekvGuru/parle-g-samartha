import sounddevice as sd
import vosk
import json
import time
import numpy as np
from resemblyzer import VoiceEncoder
import librosa
import queue

# -----------------------------
# 1. Load reference audio & compute embedding
# -----------------------------
wav_path = "my_voice_reference.wav"
wav, sr = librosa.load(wav_path, sr=16000)
encoder = VoiceEncoder()
my_embedding = encoder.embed_utterance(wav)
print("✅ Reference voice embedding computed.")
print("Embedding shape:", my_embedding.shape)

# -----------------------------
# 2. Load Vosk model
# -----------------------------
model_path = r"E:\samartha\vosk-model-en-us-0.42-gigaspeech"
model = vosk.Model(model_path)
samplerate = 16000
rec = vosk.KaldiRecognizer(model, samplerate)

# -----------------------------
# 3. Voice verification settings
# -----------------------------
THRESHOLD = 0.555        # cosine similarity threshold
chunk_duration = 0.5     # seconds per processing chunk
chunk_size = int(samplerate * chunk_duration)

audio_queue = queue.Queue()

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# -----------------------------
# 4. Callback: just queue audio
# -----------------------------
def callback(indata, frames, time_info, status):
    if status:
        print(status)
    # Copy data to avoid overwriting
    audio_queue.put(indata.copy())

# -----------------------------
# 5. Main loop: process chunks
# -----------------------------
buffer = np.zeros((0,), dtype=np.float32)
print("🎤 Listening offline... Press Ctrl+C to stop")

try:
    with sd.InputStream(samplerate=samplerate, dtype='float32', channels=1, callback=callback):
        while True:
            try:
                data = audio_queue.get(timeout=1)
                buffer = np.concatenate((buffer, data.flatten()))

                while len(buffer) >= chunk_size:
                    chunk = buffer[:chunk_size]
                    buffer = buffer[chunk_size:]

                    # Compute embedding & similarity
                    live_embedding = encoder.embed_utterance(chunk)
                    similarity = np.dot(my_embedding, live_embedding) / (np.linalg.norm(my_embedding) * np.linalg.norm(live_embedding))

                    if similarity >= THRESHOLD:
                        # Convert to int16 bytes for Vosk
                        audio_bytes = (chunk * 32768).astype(np.int16).tobytes()
                        if rec.AcceptWaveform(audio_bytes):
                            result = json.loads(rec.Result())
                            text = result.get("text", "")
                            if text:
                                print("\n🗣️ You said:", text)

            except queue.Empty:
                continue

except KeyboardInterrupt:
    print("\n🛑 Stopped")
