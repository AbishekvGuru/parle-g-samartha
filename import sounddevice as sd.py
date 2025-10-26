from audiomentations import Compose, AddGaussianNoise, PitchShift, TimeStretch, Gain
import librosa
import soundfile as sf
import glob
import os

# --- Create output folder if it doesn't exist ---
os.makedirs("augmented", exist_ok=True)

# --- Define augmentation pipeline ---
augment = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
    PitchShift(min_semitones=-2, max_semitones=2, p=0.5),
    TimeStretch(min_rate=0.9, max_rate=1.1, p=0.5),
    Gain(min_gain_db=-6, max_gain_db=6, p=0.5)
])

# --- Process all WAV files in recordings/ ---
for file in glob.glob("m4.wav"):
    y, sr = librosa.load(file, sr=16000)
    base = os.path.basename(file).replace(".wav", "")
    for i in range(6,13):  # number of augmentations per file
        out_path = f"augmented/{base}_aug{i}.wav"
        if not os.path.exists(out_path):  # skip if already exists
            y_aug = augment(samples=y, sample_rate=sr)
            sf.write(out_path, y_aug, sr)
            print(f"✅ Saved: {out_path}")
        else:
            print(f"⚠️ Skipped (already exists): {out_path}")

print("\n✨ Augmentation complete!")
