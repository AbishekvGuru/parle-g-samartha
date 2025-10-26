import os
import re
import librosa
import torch
from torch.nn.utils.rnn import pad_sequence
from datasets import Dataset
from transformers import WhisperProcessor, WhisperForConditionalGeneration, TrainingArguments, Trainer

# ---------------- CONFIG ----------------
DATA_DIR = r"E:\samartha\augmented\train"  # folder with .wav + .txt
CHUNK_DURATION = 10  # seconds per audio chunk
SAMPLING_RATE = 16000
MODEL_NAME = "openai/whisper-small"
OUTPUT_DIR = "./whisper_finetuned_me"
BATCH_SIZE = 2
NUM_EPOCHS = 10
LEARNING_RATE = 1e-5

# ---------------- LOAD PROCESSOR & MODEL ----------------
processor = WhisperProcessor.from_pretrained(MODEL_NAME)
model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)

# Move model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# ---------------- HELPER FUNCTION ----------------
def split_audio_transcript(audio_path, transcript, chunk_duration=CHUNK_DURATION, sr=SAMPLING_RATE):
    y, _ = librosa.load(audio_path, sr=sr)
    samples_per_chunk = sr * chunk_duration
    chunks = []

    transcript = re.sub(r"\s+", " ", transcript)
    words = transcript.split()

    for i in range(0, len(y), samples_per_chunk):
        chunk_audio = y[i:i+samples_per_chunk]
        start_idx = int(len(words) * i / len(y))
        end_idx = int(len(words) * min(i + samples_per_chunk, len(y)) / len(y))
        chunk_text = " ".join(words[start_idx:end_idx])

        if len(chunk_audio) == 0 or len(chunk_text.strip()) == 0:
            continue
        chunks.append({"audio": {"array": chunk_audio, "sampling_rate": sr}, "text": chunk_text})
    return chunks

# ---------------- LOAD AND SPLIT DATA ----------------
dataset_entries = []
audio_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".wav")]

for wav_file in audio_files:
    base = os.path.splitext(wav_file)[0]
    txt_file = os.path.join(DATA_DIR, base + ".txt")
    if not os.path.exists(txt_file):
        print(f"⚠️ Skipping {wav_file}, transcript missing")
        continue

    wav_path = os.path.join(DATA_DIR, wav_file)
    with open(txt_file, "r", encoding="utf-8") as f:
        transcript = f.read().strip()

    chunks = split_audio_transcript(wav_path, transcript)
    dataset_entries.extend(chunks)

print(f"✅ Total chunks prepared: {len(dataset_entries)}")

# ---------------- CREATE DATASET ----------------
dataset = Dataset.from_list(dataset_entries)

# ---------------- PREPROCESS DATA ----------------
def preprocess(batch):
    # Extract input features using WhisperProcessor
    input_features = processor(batch["audio"]["array"], sampling_rate=SAMPLING_RATE, return_tensors="pt").input_features
    labels = processor.tokenizer(batch["text"], return_tensors="pt", padding=False).input_ids

    max_len = processor.tokenizer.model_max_length
    if labels.shape[1] > max_len:
        labels = labels[:, :max_len]

    batch["input_features"] = input_features[0]
    batch["labels"] = labels[0]
    return batch

dataset = dataset.map(preprocess, remove_columns=dataset.column_names)

# ---------------- CUSTOM DATA COLLATOR ----------------
class CustomDataCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features):
        # Convert list of features to tensors
        input_features = [torch.tensor(f["input_features"], dtype=torch.float32) for f in features]
        labels = [torch.tensor(f["labels"], dtype=torch.long) for f in features]

        # Pad input features and labels
        input_features = pad_sequence(input_features, batch_first=True, padding_value=0.0)
        labels = pad_sequence(labels, batch_first=True, padding_value=self.processor.tokenizer.pad_token_id)

        return {"input_features": input_features, "labels": labels}

data_collator = CustomDataCollator(processor)

# ---------------- TRAINING ----------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=4,
    learning_rate=LEARNING_RATE,
    num_train_epochs=NUM_EPOCHS,
    fp16=torch.cuda.is_available(),  # mixed precision if GPU available
    save_strategy="epoch",
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator
)

# ---------------- START TRAINING ----------------
trainer.train()
