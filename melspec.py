import librosa
import numpy as np

# Load audio file
file_path = "D:/code/ashley/animal_emotion_proj/code/dataset/dog/sad/sad1.mp3"  # replace with your file path
signal, sr = librosa.load(file_path, sr=22050)  # sr=None to keep original

# Compute Mel Spectrogram
mel_spec = librosa.feature.melspectrogram(
    y=signal,
    sr=sr,
    n_fft=2048,
    hop_length=512,
    n_mels=128
)

# Convert to log scale (dB)
mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

print("Mel Spectrogram shape:", mel_spec_db.shape)
import matplotlib.pyplot as plt
import librosa.display

plt.figure(figsize=(10, 4))
librosa.display.specshow(mel_spec_db, sr=sr, hop_length=512, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title("Mel Spectrogram")
plt.tight_layout()
plt.show()