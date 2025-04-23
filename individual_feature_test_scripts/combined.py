import librosa
import numpy as np
from sklearn.preprocessing import LabelEncoder


def extract_tempo(wav_path: str) -> float:
    """
    Extract the tempo (in BPM) from a .wav file.
    """
    y, sr = librosa.load(wav_path)

    # Correct call for librosa 0.11.0
    tempo = float(librosa.feature.tempo(y=y, sr=sr, aggregate=None).mean())

    return tempo


def extract_time_signature(wav_path: str) -> str:
    """
    Estimate the time signature from a .wav file.
    """
    y, sr = librosa.load(wav_path)

    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    intervals = np.diff(beat_times)
    mean_interval = np.mean(intervals)

    if mean_interval < 0.7:
        return "4/4"
    elif mean_interval < 1.0:
        return "3/4"
    else:
        return "6/8"


def detect_key(wav_file_path: str) -> str:
    """
    Detect the musical key of a .wav file.
    """
    # Load the audio file
    y, sr = librosa.load(wav_file_path)

    # Extract the chroma feature from the audio signal
    chroma = librosa.feature.chroma_cens(y=y, sr=sr)

    # Compute the average chroma over time
    chroma_avg = np.mean(chroma, axis=1)

    # Define the chroma labels
    chroma_labels = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

    # Find the peak in the chroma feature to detect the key
    key_index = np.argmax(chroma_avg)

    # Get the corresponding musical key
    detected_key = chroma_labels[key_index]

    return detected_key


def extract_mode(wav_file_path):
    # Load the audio file
    y, sr = librosa.load(wav_file_path)

    # Extract the chroma feature (Chroma Energy Normalized)
    chroma = librosa.feature.chroma_cens(y=y, sr=sr)

    # Compute the average chroma across all frames (columns)
    chroma_avg = np.mean(chroma, axis=1)

    # Define the chroma labels for musical notes
    chroma_labels = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

    # Normalize the chroma to avoid bias towards higher values
    chroma_avg = chroma_avg / np.sum(chroma_avg)

    # Mapping to major and minor modes
    # Simplified model based on the harmonic distribution of notes for major and minor scales:
    # Major scale intervals: [0, 2, 4, 5, 7, 9, 11]
    # Minor scale intervals: [0, 2, 3, 5, 7, 8, 10]
    major_mode_intervals = [0, 2, 4, 5, 7, 9, 11]  # Notes in a major scale
    minor_mode_intervals = [0, 2, 3, 5, 7, 8, 10]  # Notes in a natural minor scale

    # Find the most common note in the chroma feature (index of the highest chroma value)
    predominant_note_index = np.argmax(chroma_avg)

    # Calculate the distance between the chroma pattern and the major/minor scale
    chroma_dist_major = sum(
        [abs(predominant_note_index - note) for note in major_mode_intervals]
    )
    chroma_dist_minor = sum(
        [abs(predominant_note_index - note) for note in minor_mode_intervals]
    )

    # Return the mode based on the closest match
    if chroma_dist_major < chroma_dist_minor:
        return "Major"
    else:
        return "Minor"


def get_duration_ms(wav_file_path):
    # Load the audio file
    y, sr = librosa.load(wav_file_path)

    # Calculate the duration in seconds
    duration_sec = librosa.get_duration(y=y, sr=sr)

    # Convert duration to milliseconds
    duration_ms = duration_sec * 1000  # 1 second = 1000 milliseconds

    return duration_ms


def main():
    wav_file = "./downloaded_playlist/Astrality, Aiko - Restart My Heart.wav"  # Replace this with your file path

    bpm = extract_tempo(wav_file)
    print(f"The estimated tempo is: {bpm:.2f} BPM")

    time_sig = extract_time_signature(wav_file)
    print(f"The estimated time signature is: {time_sig}")

    key = detect_key(wav_file)
    print(f"The detected key is: {key}")

    mode = extract_mode(wav_file)
    print(f"The mode of the audio is: {mode}")

    duration = get_duration_ms(wav_file)
    print(f"The duration of the audio is: {duration:.2f} ms")


if __name__ == "__main__":
    main()
