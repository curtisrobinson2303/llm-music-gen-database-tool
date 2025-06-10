# LLM Music Generation Database Tool

## Overview

The **LLM Music Generation Database Tool** is an end-to-end pipeline for collecting, processing, and analyzing music data. It supports extracting metadata and audio features from Spotify playlists, converting audio formats, extracting chord progressions, visualizing music structure, and generating datasets for training language models or other machine learning applications.

---

## Features

- **Spotify Playlist Download:** Download tracks from a Spotify playlist using [spotdl](https://github.com/spotDL/spotify-downloader).
- **Audio Conversion:** Convert MP3 files to WAV format in batch.
- **Audio Feature Extraction:** Extract tempo, key, mode, time signature, duration, danceability, energy, and loudness from WAV files.
- **Chord Extraction:** Extract chord progressions from WAV files using harmonic profiling.
- **Chord Visualization:** Visualize chord progressions over time.
- **MIDI/CSV/JSON/Text Conversion:** Convert between chord CSV, MIDI, JSON, and plain text representations.
- **Dataset Generation:** Combine input (audio features) and output (chord progressions) into structured JSON and CSV datasets for ML training.
- **Command-Line Interface:** Modular CLI for all major operations.

---

## Project Structure

```
.
├── music_tool.py                # Main script with all pipeline logic and CLI
├── average_duration.py
├── credentials.json             # Spotify API credentials (not tracked by git)
├── input_output_dataset.json    # Combined input/output dataset (generated)
├── music_training_dataset       # Final training dataset (CSV, generated)
├── setup_env.py
├── requirements.txt
├── README.md
├── chord_visualizations/        # Saved chord progression plots
├── converted_wavs/              # WAV files converted from MP3s or downloaded
├── input_pair_json_data/        # Extracted audio features (JSON)
├── output_pair_json_data/       # Chord progressions (JSON)
├── processed_tracks_csv/        # Chord progressions (CSV)
├── processed_tracks_midi/       # Chord progressions (MIDI)
├── tracks_wav/                  # Downloaded WAV files from Spotify
└── ...
```

---

## Setup

### 1. Install Dependencies

Install required libraries (see `requirements.txt` for full list):

```bash
pip install spotipy pandas mido librosa matplotlib essentia pydub scikit-learn
```

You also need [spotdl](https://github.com/spotDL/spotify-downloader) for downloading Spotify tracks:

```bash
pip install spotdl
```

### 2. Set Up Spotify API Credentials

- Create a Spotify Developer account.
- Register an app and obtain `client_id` and `client_secret`.
- Store credentials in `credentials.json`:

```json
{
    "client_id": "your_client_id",
    "client_secret": "your_client_secret"
}
```

---

## Usage

All functionality is accessed via the command-line interface in [`music_tool.py`](music_tool.py):

### General Syntax

```bash
python3 music_tool.py <command> [options]
```

### Main Commands

#### 1. Convert MP3s to WAV

```bash
python3 music_tool.py convert --input_folder <mp3_folder> --output_folder <wav_folder>
```

#### 2. Extract Chords from WAVs

```bash
python3 music_tool.py extract --input_folder <wav_folder> --output_folder <csv_folder>
```

#### 3. Visualize Chord Progression

```bash
python3 music_tool.py visualize --csv_file <chord_csv> --save_folder <img_folder> [--no_show]
```

#### 4. Convert CSV to MIDI

```bash
python3 music_tool.py csv2midi --csv_file <csv_file> --output_folder <midi_folder>
# or batch:
python3 music_tool.py csv2midi --input_folder <csv_folder> --output_folder <midi_folder>
```

#### 5. Convert MIDI to Text

```bash
python3 music_tool.py midi2txt --midi_file <midi_file> --output_folder <txt_folder>
```

#### 6. Convert MIDI to JSON

```bash
python3 music_tool.py midi2json --midi_file <midi_file> --output_folder <json_folder>
# or batch:
python3 music_tool.py midi2json --input_folder <midi_folder> --output_folder <json_folder>
```

#### 7. Convert MIDI to CSV (Chord Progression)

```bash
python3 music_tool.py midi2csv --midi_file <midi_file> --output_folder <csv_folder>
# or batch:
python3 music_tool.py midi2csv --input_folder <midi_folder> --output_folder <csv_folder>
```

#### 8. Full Database Generation Pipeline

**From Spotify Playlist:**

```bash
python3 music_tool.py databasegen --input_url "<spotify_playlist_url>" --credentials_path "credentials.json"
```

**From Local Downloaded WAVs:**

```bash
python3 music_tool.py databasegen --tracks_dir <wav_folder> --credentials_path "credentials.json"
```

---

## Pipeline Details

### Database Generation Steps

1. **Download WAVs:** Download all tracks from a Spotify playlist or use your own WAVs.
2. **Extract Audio Features:** Extract tempo, key, mode, time signature, duration, danceability, energy, and loudness for each WAV (`input_pair_json_data/`).
3. **Chord Extraction:** Extract chord progressions from WAVs and save as CSV (`processed_tracks_csv/`).
4. **CSV to MIDI:** Convert chord CSVs to MIDI files (`processed_tracks_midi/`).
5. **MIDI to JSON:** Convert MIDI files to JSON (`output_pair_json_data/`).
6. **Pair Input/Output:** Combine input (audio features) and output (chord progression) JSONs into a single dataset (`input_output_dataset.json`).
7. **Format for ML:** Convert the paired JSON into a CSV suitable for ML training (`music_training_dataset`).

---

## Example Workflow

```bash
# 1. Download playlist and generate dataset
python3 music_tool.py databasegen --input_url "<spotify_playlist_url>" --credentials_path "credentials.json"

# 2. Or, process your own WAVs
python3 music_tool.py databasegen --tracks_dir tracks_wav --credentials_path "credentials.json"
```

---

## Notes

- Ensure your Spotify app settings allow `http://localhost:8080` as a redirect URI.
- The Spotify API may enforce rate limits; wait before making frequent requests.
- All intermediate and final datasets are saved in their respective folders.

---

## Advanced: Individual Functionality

You can use any of the CLI commands independently for custom workflows (e.g., just converting MIDI to CSV, or visualizing chords).

---

## Contributing

This tool is actively being developed. Contributions and suggestions are welcome!

---

## License

See `LICENSE` file (if present) for details.

