import numpy as np, librosa
from resemblyzer import VoiceEncoder

encoder = VoiceEncoder()
samplerate = 16000

def get_voiceprint(paths):
    embeds = []
    for p in paths:
        wav, _ = librosa.load(p, sr=samplerate)
        embeds.append(encoder.embed_utterance(wav))
    return np.mean(embeds, axis=0)

reference_emb = get_voiceprint([
    "m1.wav",
    "m2.wav",
    "m3.wav"
])
np.save("my_voice_reference.npy", reference_emb)
print("✅ Saved stable voiceprint!")
