import os
import json
import numpy as np
import librosa
from datetime import datetime
import time
from datetime import timedelta


def extract_tempo(wav_path: str) -> float:
    y, sr = librosa.load(wav_path)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    return float(tempo)


def extract_time_signature(wav_path: str) -> str:
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


def extract_key(wav_file_path: str) -> str:
    y, sr = librosa.load(wav_file_path)
    chroma = librosa.feature.chroma_cens(y=y, sr=sr)
    chroma_avg = np.mean(chroma, axis=1)
    chroma_labels = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    key_index = np.argmax(chroma_avg)
    detected_key = chroma_labels[int(key_index)]
    return str(detected_key)


def extract_mode(wav_file_path: str) -> str:
    y, sr = librosa.load(wav_file_path)
    chroma = librosa.feature.chroma_cens(y=y, sr=sr)
    chroma_avg = np.mean(chroma, axis=1)
    chroma_avg = chroma_avg / np.sum(chroma_avg)

    major_mode_intervals = [0, 2, 4, 5, 7, 9, 11]
    minor_mode_intervals = [0, 2, 3, 5, 7, 8, 10]

    predominant_note_index = np.argmax(chroma_avg)

    chroma_dist_major = sum(
        abs(predominant_note_index - note) for note in major_mode_intervals
    )
    chroma_dist_minor = sum(
        abs(predominant_note_index - note) for note in minor_mode_intervals
    )

    return "Major" if chroma_dist_major < chroma_dist_minor else "Minor"


def extract_duration(wav_file_path: str) -> float:
    y, sr = librosa.load(wav_file_path)
    duration_sec = librosa.get_duration(y=y, sr=sr)
    return float(duration_sec * 1000)


def extract_danceability(wav_path: str) -> float:
    y, sr = librosa.load(wav_path)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    return float(tempo)


def extract_energy(wav_path: str) -> float:
    y, sr = librosa.load(wav_path)
    rms = librosa.feature.rms(y=y)[0]
    avg_energy = np.mean(rms)
    return float(avg_energy)


def extract_loudness(wav_path: str) -> float:
    y, sr = librosa.load(wav_path)
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S += 1e-10
    log_S = librosa.power_to_db(S, ref=1.0)
    mel_freqs = librosa.mel_frequencies(n_mels=S.shape[0])
    loudness = np.mean(librosa.perceptual_weighting(log_S, frequencies=mel_freqs))
    return float(loudness)


# --- Smart Processing + Visual Progress Bar ---
def process_directory_to_json(directory_path: str, output_json_path: str):
    results = {}

    if not os.path.isdir(directory_path):
        print(f"Error: Directory {directory_path} does not exist.")
        return

    wav_files = [f for f in os.listdir(directory_path) if f.lower().endswith(".wav")]

    if not wav_files:
        print(f"No .wav files found in {directory_path}.")
        return

    # Load existing results if JSON already exists
    if os.path.exists(output_json_path):
        try:
            with open(output_json_path, "r") as f:
                results = json.load(f)
            print(
                f"Loaded existing JSON with {len(results)} entries. Will skip already processed files."
            )
        except Exception as e:
            print(f"Warning: Failed to load existing JSON. Starting fresh. Error: {e}")

    files_to_process = [f for f in wav_files if f not in results]

    if not files_to_process:
        print("All .wav files are already processed. Nothing to do!")
        return

    print(f"Found {len(files_to_process)} new files to process...\n")

    start_time = time.time()
    total_files = len(files_to_process)

    for idx, filename in enumerate(files_to_process, start=1):
        wav_path = os.path.join(directory_path, filename)
        try:
            song_data = {
                "tempo": extract_tempo(wav_path),
                "time_signature": extract_time_signature(wav_path),
                "key": extract_key(wav_path),
                "mode": extract_mode(wav_path),
                "duration_ms": extract_duration(wav_path),
                "danceability": extract_danceability(wav_path),
                "energy": extract_energy(wav_path),
                "loudness": extract_loudness(wav_path),
            }
            results[filename] = song_data
        except Exception as e:
            print(f"\nError processing {filename}: {e}")

        # Progress bar and ETA calculation
        elapsed = time.time() - start_time
        avg_time_per_file = elapsed / idx
        remaining_files = total_files - idx
        eta_seconds = remaining_files * avg_time_per_file
        eta = str(timedelta(seconds=int(eta_seconds)))

        progress = (idx / total_files) * 100
        bar_length = 40  # Length of progress bar
        filled_length = int(progress / 100 * bar_length)
        bar = "█" * filled_length + "-" * (bar_length - filled_length)
        print(f"\r[{bar}] {progress:.2f}% | ETA: {eta}", end="")

    print("\n\nProcessing complete.")

    output_dir = os.path.dirname(output_json_path)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")
        except Exception as e:
            print(f"Failed to create output directory {output_dir}: {e}")
            return

    try:
        with open(output_json_path, "w") as f:
            json.dump(results, f, indent=4)
        print(
            f"Updated JSON saved to {output_json_path} with {len(results)} total entries."
        )
    except Exception as e:
        print(f"Failed to write JSON file: {e}")


def main():
    # Setup paths
    input_directory = "downloaded_playlist"
    sample_wav = None

    try:
        wav_files = [
            f for f in os.listdir(input_directory) if f.lower().endswith(".wav")
        ]
        if not wav_files:
            print(f"No .wav files found in {input_directory}. Exiting.")
            return
        sample_wav = os.path.join(input_directory, wav_files[0])
    except Exception as e:
        print(f"Error accessing {input_directory}: {e}")
        return

    # Dynamic output filename
    output_dir = "output_data"
    output_json_filename = f"{os.path.basename(input_directory)}_features.json"
    output_json_path = os.path.join(output_dir, output_json_filename)

    print("Testing individual feature extraction functions:")
    try:
        print(f"Sample file: {sample_wav}")
        print(f"Tempo: {extract_tempo(sample_wav)} BPM")
        print(f"Time Signature: {extract_time_signature(sample_wav)}")
        print(f"Key: {extract_key(sample_wav)}")
        print(f"Mode: {extract_mode(sample_wav)}")
        print(f"Duration: {extract_duration(sample_wav)} ms")
        print(f"Danceability: {extract_danceability(sample_wav)} (based on tempo)")
        print(f"Energy: {extract_energy(sample_wav)}")
        print(f"Loudness: {extract_loudness(sample_wav)}")
    except Exception as e:
        print(f"Error extracting features from sample file: {e}")
        return

    print("\nTesting full directory processing to JSON (with smart skipping):")
    process_directory_to_json(input_directory, output_json_path)


if __name__ == "__main__":
    main()
