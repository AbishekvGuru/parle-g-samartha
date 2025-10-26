import sounddevice as sd
import queue
import json
import numpy as np
import librosa
import vosk
from resemblyzer import VoiceEncoder
from numpy.linalg import norm

# ---------------- CONFIG ----------------
samplerate = 16000
chunk_duration = 1.5   # seconds per processing chunk
chunk_size = int(samplerate * chunk_duration)
threshold = 0.65      # voice similarity threshold (tune 0.6–0.8 based on mic)
encoder = VoiceEncoder()

# ---------------- LOAD VOICE REFERENCE ----------------
# Option 1: Load existing embedding
try:
    reference_emb = np.load("my_voice_reference.npy")
    print("✅ Loaded saved voice reference embedding.")
except FileNotFoundError:
    # Option 2: Compute from reference WAV if embedding missing
    wav_path = "my_voice_reference.wav"
    wav, sr = librosa.load(wav_path, sr=samplerate)
    reference_emb = encoder.embed_utterance(wav)
    np.save("my_voice_reference.npy", reference_emb)
    print("✅ Computed and saved new voice reference embedding.")

print("Embedding shape:", reference_emb.shape)

# ---------------- LOAD VOSK MODEL ----------------
model_path = r"E:\samartha\vosk-model-en-us-0.42-gigaspeech"
print("Loading Vosk model...")
model = vosk.Model(model_path)
rec = vosk.KaldiRecognizer(model, samplerate)
print("✅ Model loaded.\n")

# ---------------- AUDIO STREAM SETUP ----------------
q_audio = queue.Queue()

def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    q_audio.put(indata.copy())

# ---------------- SPEAKER VERIFICATION ----------------
def compute_similarity(chunk):
    """Compute cosine similarity between reference and live chunk."""
    live_emb = encoder.embed_utterance(chunk)
    sim = np.dot(reference_emb, live_emb) / (norm(reference_emb) * norm(live_emb))
    return sim

# ---------------- PREPROCESS AUDIO ----------------
def preprocess_audio(chunk):
    """Trim silence and return active voice segments."""
    intervals = librosa.effects.split(chunk, top_db=25)
    parts = [chunk[s:e] for s, e in intervals]
    return np.concatenate(parts) if parts else None

# ---------------- MAIN LOOP ----------------
print("🎤 Listening... Press Ctrl+C to stop\n")

buffer = np.zeros(0, dtype=np.float32)
last_partial_text = ""
last_final_text = ""

try:
    with sd.InputStream(samplerate=samplerate, channels=1, dtype='float32', callback=audio_callback):
        while True:
            data = q_audio.get()
            buffer = np.concatenate((buffer, data.flatten()))

            # Process chunks
            if len(buffer) >= chunk_size:
                chunk = buffer[:chunk_size]
                buffer = buffer[chunk_size:]

                processed = preprocess_audio(chunk)
                if processed is None or len(processed) < 1000:
                    continue

                # Compute similarity
                similarity = compute_similarity(processed)
                print(f"Voice similarity: {similarity:.2f}", end="\r")

                if similarity < threshold:
                    continue  # skip if not your voice

                # Perform speech recognition
                audio_bytes = (processed * 32768).astype(np.int16).tobytes()

                if rec.AcceptWaveform(audio_bytes):
                    result = json.loads(rec.Result())
                    text = result.get("text", "").strip()
                    if text and text != last_final_text:
                        print(f"\n🗣️ You said: {text}")
                        last_final_text = text
                        last_partial_text = ""
                else:
                    partial = json.loads(rec.PartialResult())
                    ptext = partial.get("partial", "").strip()
                    if ptext and ptext != last_partial_text:
                        print(f"... {ptext}", end="\r")
                        last_partial_text = ptext

except KeyboardInterrupt:
    print("\n🛑 Stopped.")
