import numpy as np
import librosa
import essentia.standard as es
from pydub import AudioSegment
import os


def extract_audio_features(file_path):
    # --- Load Audio ---
    y, sr = librosa.load(file_path)
    audio = es.MonoLoader(filename=file_path)()

    # --- Feature Extraction ---
    # Duration (ms)
    duration_ms = len(AudioSegment.from_wav(file_path))

    # Tempo
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

    # Energy (RMS)
    rms = librosa.feature.rms(y=y)[0].mean()

    # Loudness (dB)
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    loudness = librosa.power_to_db(S, ref=np.max).mean()

    # Key + Mode
    key, mode, strength = es.KeyExtractor()(audio)

    # Acousticness (Essentia's feature)
    acousticness = es.Acousticness()(audio)

    # Danceability approximation via tempo
    danceability = tempo / 250 if tempo < 250 else 1.0  # Normalize to [0,1]

    # Instrumentalness approx via energy in vocal track
    instrumentalness = 0.0
    try:
        os.system(
            f"spleeter separate -i {file_path} -p spleeter:2stems -o spleeter_output"
        )
        vocal_path = os.path.join(
            "spleeter_output",
            os.path.basename(file_path).replace(".wav", ""),
            "vocals.wav",
        )
        vocals, _ = librosa.load(vocal_path)
        vocals_rms = librosa.feature.rms(y=vocals)[0].mean()
        instrumentalness = 1.0 if vocals_rms < 0.01 else 0.0
    except Exception:
        pass

    # Speechiness approximation
    speechiness = (
        0.0  # Placeholder (actual needs classifier or voice activity detection)
    )

    # Liveness approximation (e.g., high-frequency content)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0].mean()
    liveness = spectral_centroid / sr

    # Time Signature (Madmom alternative skipped for speed)
    time_signature = 4  # Default to 4/4, override if madmom is installed

    # Valence approximation (needs classifier — stub here)
    valence = 0.5  # Placeholder

    # --- Concatenate features ---
    features = np.array(
        [
            acousticness,
            danceability,
            duration_ms,
            rms,  # energy
            instrumentalness,
            key_to_number(key),
            liveness,
            loudness,
            1 if mode == "major" else 0,
            speechiness,
            tempo,
            time_signature,
            valence,
        ]
    )

    print("Extracted Feature Vector:\n", features)
    return features


# --- Helper: Convert key string to numerical pitch class ---
def key_to_number(key):
    keys = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    return keys.index(key) if key in keys else -1


# --- Example Usage ---
if __name__ == "__main__":
    extract_audio_features("test-song.wav")
