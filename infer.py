import torch
import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# --- Load fine-tuned model ---
MODEL_DIR = "./whisper_finetuned_me"
processor = WhisperProcessor.from_pretrained(MODEL_DIR)
model = WhisperForConditionalGeneration.from_pretrained(MODEL_DIR)
model.to("cuda" if torch.cuda.is_available() else "cpu")

# --- Load audio file ---
AUDIO_PATH = r"E:\samartha\m2.wav"
audio, sr = librosa.load(AUDIO_PATH, sr=16000)

# --- Prepare features ---
inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
input_features = inputs.input_features.to(model.device)

# --- Generate transcription ---
with torch.no_grad():
    predicted_ids = model.generate(input_features)

# --- Decode tokens to text ---
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
print("\n📝 Transcription:\n", transcription)
