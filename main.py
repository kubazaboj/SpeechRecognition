import os
import numpy as np
import librosa
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

hop_length = 512
n_mfcc = 13
X = []
y = []
original_labels = [0, 1, 1, 0] #1 = male, 0 = female
def get_augmented_versions(original_wave, original_sr):
    """Returns a list containing the original and 3 augmented versions."""
    versions = [original_wave, librosa.effects.pitch_shift(y=original_wave, sr=original_sr, n_steps=-2),
                librosa.effects.time_stretch(y=original_wave, rate=1.2)]

    #Add Noise
    noise = np.random.randn(len(original_wave))
    versions.append(original_wave + 0.005 * noise)

    return versions

files = sorted(os.listdir("example_audio"))

for i, file in enumerate(files):
    print(f"Processing {file}...")
    path = os.path.join("example_audio", file)
    wave, sr = librosa.load(path)

    # Get original + 3 new versions
    augmented_waves = get_augmented_versions(wave, sr)

    # Loop through all versions (original + augmented)
    for augmented_wave in augmented_waves:
        # Extract Features
        mfccs = librosa.feature.mfcc(y=augmented_wave, sr=sr, hop_length=hop_length, n_mfcc=n_mfcc)
        mfccs_avg = np.mean(mfccs, axis=1)

        X.append(mfccs_avg)

        # IMPORTANT: Reuse the same label for all augmented versions of this file
        y.append(original_labels[i])

X = np.array(X)
y = np.array(y)

print(f"New Data Shape: {X.shape}")
print(f"New Label Shape: {y.shape}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=42)


model = SVC(kernel='linear', C=1.0, gamma='auto')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
cm_disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["female", "male"])

if __name__ == "__main__":
    print(cm)
    print("Testing accuracy:", model.score(X_test, y_test))

