import librosa
import numpy as np


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


# Example usage
wav_file_path = "./downloaded_playlist/Astrality, Aiko - Restart My Heart.wav"
mode = extract_mode(wav_file_path)
print(f"The mode of the audio is: {mode}")
