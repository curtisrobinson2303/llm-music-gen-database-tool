# LLM Music Generation Database Tool

## Overview

The **LLM Music Generation Database Tool** is designed to collect, process, and analyze music metadata and audio data from Spotify. It facilitates the creation of structured datasets for training language models with music-related content.

## Features

- **Playlist Data Extraction:** Fetches song metadata (ID, name) from a Spotify playlist.
- **Audio Analysis:** Retrieves detailed audio analysis data for each track using Spotify's API.
- **Database Generation:** Combines metadata and audio analysis results into structured datasets.
- **Customizable & Modular:** Designed with a modular approach to allow easy customization and expansion.

## Components

### 1. **Spotify API Integration**

- Utilizes `spotipy` to authenticate and fetch data from Spotify.
- Requires Spotify credentials stored in `credentials.json`.

### 2. **Input Data Processing**

- Extracts song metadata (ID, name) from a playlist and stores it in a `.csv` file.
- Uses `generate_input_csv()` and `process_input_metadata()` to organize data.

### 3. **Audio Analysis Pipeline**

- Uses the `audio_analysis()` API endpoint to fetch detailed analysis for each track.
- Implemented in `test_audio_analysis.py` for testing individual tracks.

### 4. **Project Files**

- `music_tool.py` - Main script that orchestrates the database generation process.
- `test_audio_analysis.py` - Standalone script for debugging Spotify's audio analysis API.
- `.gitignore` - Excludes sensitive files (e.g., `credentials.json`), logs, and temporary data.
- `README.md` - Project documentation.

## Setup

### 1. Install Dependencies

Run the following command to install required libraries:

```bash
pip install spotipy pandas
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

### 3. Run the Tool

To generate a dataset from a Spotify playlist:

```bash
python3 music_tool.py databasegen --input_url "<spotify_playlist_url>" --credentials_path "credentials.json"
```

To test audio analysis for a single track:

```bash
python3 test_audio_analysis.py
```

## Notes

- Ensure that your Spotify app settings allow `http://localhost:8080` as a redirect URI.
- The Spotify API may enforce rate limits; wait before making frequent requests.

## Future Enhancements

- Implement audio feature extraction using local MP3 files.
- Enhance dataset structure with additional music attributes.
- Support for batch audio processing.

---

This tool is actively being developed. Contributions and suggestions are welcome!

