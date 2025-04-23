import librosa
import numpy as np
from sklearn.preprocessing import StandardScaler


def extract_valence(wav_file_path):
    # Load the audio file
    y, sr = librosa.load(wav_file_path)

    # Extract chroma features (a good indicator for the harmonic content)
    chroma = librosa.feature.chroma_cens(y=y, sr=sr)

    # Extract spectral contrast features (also related to mood and timbre)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)

    # Extract tempo and rhythmic information (important for mood)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)

    # Compute the mean of the chroma and spectral contrast features
    chroma_mean = np.mean(chroma, axis=1)
    spectral_contrast_mean = np.mean(spectral_contrast, axis=1)

    # Ensure tempo_feature is a 1D array with the tempo value
    tempo_feature = np.array([tempo])  # Convert the tempo scalar to a 1D array

    # Combine features into a single feature vector
    features = np.concatenate((chroma_mean, spectral_contrast_mean, tempo_feature))

    # Standardize the features to have zero mean and unit variance (for machine learning)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features.reshape(-1, 1)).flatten()

    # For now, we'll create a simple heuristic that uses the features to approximate valence.
    # Note: A real model would require labeled training data, but here we'll just
    # use some basic logic based on features.

    # Simple heuristic for valence prediction (for example, higher tempo might indicate a more positive mood)
    valence = (
        np.mean(features_scaled[:12])
        + np.mean(features_scaled[12:24])
        + features_scaled[24]
    )

    # Normalizing the result to give a score between -1 and 1 (just a basic range)
    valence_score = np.clip(valence, -1, 1)

    return valence_score


# Example usage
wav_file_path = "./downloaded_playlist/Astrality, Aiko - Restart My Heart.wav"  # Replace with the correct path to your audio file
valence = extract_valence(wav_file_path)
print(f"The valence of the audio is: {valence:.2f}")
