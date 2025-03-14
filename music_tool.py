#!/usr/bin/env python
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import essentia.standard as es
from pydub import AudioSegment
import mido
import json

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import time

from spotipy.oauth2 import SpotifyOAuth

# from mido import Message, MidiFile, MidiTrack


# *****************************************************
#
#   convert_mp3_to_wav
#
#   Brief: givena  .mp3 file convert it to a .wav file
#
# *****************************************************
def convert_mp3_to_wav(input_file, output_folder="converted_wavs"):
    """
    Converts an MP3 file to WAV format.
    Saves the converted file in the specified output folder.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    filename = os.path.splitext(os.path.basename(input_file))[0]
    output_file = os.path.join(output_folder, f"{filename}.wav")
    try:
        audio = AudioSegment.from_mp3(input_file)
        audio.export(output_file, format="wav")
        print(f"✅ Converted: {input_file} → {output_file}")
        return output_file
    except Exception as e:
        print(f"❌ Error converting {input_file}: {e}")
        return None


# *****************************************************
#
#   convert_folder
#
#   Brief: Given a folder of .mp3 files convert all of them to a folder containing .wav files
#
# *****************************************************
def convert_folder(input_folder, output_folder="converted_wavs"):
    """
    Converts all MP3 files in the given input folder to WAV format.
    """
    if not os.path.exists(input_folder):
        print(f"Input folder '{input_folder}' does not exist.")
        return
    files = os.listdir(input_folder)
    mp3_files = [f for f in files if f.lower().endswith(".mp3")]
    if not mp3_files:
        print("No MP3 files found in the input folder.")
        return
    print(
        f"Found {len(mp3_files)} MP3 file(s) in '{input_folder}'. Starting conversion..."
    )
    for file in mp3_files:
        input_file_path = os.path.join(input_folder, file)
        convert_mp3_to_wav(input_file_path, output_folder)
    print("Conversion complete.")


# *****************************************************
#
#   extract_chords_from_wav
#
#   Brief: Given a .wav file extract the chords using harmonic profiling and output a .csv file containing the timestamp and chord
#
# *****************************************************
def extract_chords_from_wav(wav_file, output_folder="chord_data"):
    """
    Extracts chord progressions from a WAV file and saves the output as a CSV.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    loader = es.MonoLoader(filename=wav_file)
    audio = loader()
    sample_rate = 44100
    total_duration = len(audio) / sample_rate
    print(f"Loaded audio duration for '{wav_file}': {total_duration:.2f} seconds")
    # HPCP extraction setup
    frame_size = 2048
    hop_size = 512
    frame_generator = es.FrameGenerator(
        audio, frameSize=frame_size, hopSize=hop_size, startFromZero=True
    )
    windowing = es.Windowing(type="hann")
    spectrum = es.Spectrum()
    spectral_peaks = es.SpectralPeaks()
    hpcp_algo = es.HPCP()
    hpcp_features = []
    for frame in frame_generator:
        spec = spectrum(windowing(frame))
        freqs, mags = spectral_peaks(spec)
        hpcp_vector = hpcp_algo(freqs, mags)
        hpcp_features.append([float(x) for x in hpcp_vector])
    print(f"Extracted HPCP features for {len(hpcp_features)} frames from '{wav_file}'.")
    # Beat tracking using RhythmExtractor2013
    rhythm_extractor = es.RhythmExtractor2013(method="multifeature")
    bpm, beat_positions, beats_confidence, _, _ = rhythm_extractor(audio)
    beat_positions = [float(b) for b in beat_positions]
    print(f"Detected {len(beat_positions)} beats in '{wav_file}'.")
    # Chord detection using ChordsDetectionBeats
    chord_extractor = es.ChordsDetectionBeats()
    try:
        chords, strengths = chord_extractor(hpcp_features, beat_positions)
    except Exception as e:
        print(f"Error in chord extraction for '{wav_file}': {e}")
        return None
    print(f"Extracted {len(chords)} chords from '{wav_file}'.")
    chord_data = []
    for i, beat in enumerate(beat_positions):
        if i < len(chords):
            chord_data.append({"Timestamp": round(beat, 2), "Chord": chords[i]})
    df = pd.DataFrame(chord_data)
    output_csv = os.path.join(
        output_folder, os.path.basename(wav_file).replace(".wav", "_chords.csv")
    )
    df.to_csv(output_csv, index=False)
    print(f"Chords extracted and saved to: {output_csv}\n")
    return df


# *****************************************************
#
#   process_wav_folder
#
#   Brief: Given a folder containing .wav files process each wav file for chord extraction
#
# *****************************************************
def process_wav_folder(input_folder="converted_wavs", output_folder="chord_data"):
    """
    Processes all WAV files in the input folder, extracting chord progressions.
    """
    if not os.path.exists(input_folder):
        print(f"Input folder '{input_folder}' does not exist.")
        return
    wav_files = [f for f in os.listdir(input_folder) if f.lower().endswith(".wav")]
    if not wav_files:
        print(f"No .wav files found in '{input_folder}'.")
        return
    print(f"Found {len(wav_files)} WAV file(s) in '{input_folder}'. Processing...")
    for wav_file in wav_files:
        full_path = os.path.join(input_folder, wav_file)
        extract_chords_from_wav(full_path, output_folder)


# *****************************************************
#
#   visualize_chords_on_y_axis
#
#   Brief: Given a .csv file and the save folder location, visualize the chord progression on the y-axis
#
# *****************************************************
def visualize_chords_on_y_axis(
    csv_file, save_folder="chord_visualizations", show_plot=True
):
    """
    Visualizes the chord progression with time on the x-axis and chords on the y-axis.
    """
    df = pd.read_csv(csv_file)
    unique_chords = sorted(df["Chord"].unique())
    chord_to_num = {chord: i for i, chord in enumerate(unique_chords)}
    df["ChordNum"] = df["Chord"].map(chord_to_num)
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.scatter(df["Timestamp"], df["ChordNum"], color="blue", s=50)
    ax.set_yticks(list(chord_to_num.values()))
    ax.set_yticklabels(list(chord_to_num.keys()))
    ax.set_xlabel("Time (seconds)")
    ax.set_title("Chord Progression Over Time")
    ax.grid(axis="x", linestyle="--", alpha=0.7)
    plt.tight_layout()
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    base_name = os.path.splitext(os.path.basename(csv_file))[0]
    output_file = os.path.join(save_folder, f"{base_name}.png")
    plt.savefig(output_file)
    print(f"Plot saved to: {output_file}")
    if show_plot:
        plt.show()
    else:
        plt.close()


# *****************************************************
#
#   chord_to_midi_notes
#
#   Brief: Given a chord label, convert chord data to MIDI note numbers.
#
# *****************************************************
def chord_to_midi_notes(chord):
    """
    Converts a chord label (e.g., 'C', 'Cm', 'G', 'Em') to a list of MIDI note numbers.
    Assumes a simple interpretation: major chords use intervals [0,4,7], minor chords use [0,3,7].
    The root note is mapped to the 4th octave (C4=60).
    """
    chord = chord.strip()
    is_minor = chord.endswith("m")
    if is_minor:
        root = chord[:-1]
        intervals = [0, 3, 7]
    else:
        root = chord
        intervals = [0, 4, 7]
    note_to_midi = {
        "C": 60,
        "C#": 61,
        "Db": 61,
        "D": 62,
        "D#": 63,
        "Eb": 63,
        "E": 64,
        "F": 65,
        "F#": 66,
        "Gb": 66,
        "G": 67,
        "G#": 68,
        "Ab": 68,
        "A": 69,
        "A#": 70,
        "Bb": 70,
        "B": 71,
    }
    if root in note_to_midi:
        base = note_to_midi[root]
    else:
        base = 60  # default to C if unknown
    return [base + interval for interval in intervals]


# *****************************************************
#
#   detect_chord
#
#   Brief: Given a list of MIDI note numbers, detect the chord as a simple major or minor triad.
#
# *****************************************************
def detect_chord(notes):
    """
    Detects the chord from a list of MIDI note numbers by matching against major and minor triad templates.
    Returns the chord label as a string (e.g., "Ab", "Abm").
    """
    mods = sorted(set([n % 12 for n in notes]))
    major_template = [0, 4, 7]
    minor_template = [0, 3, 7]
    note_names = {
        0: "C",
        1: "C#",
        2: "D",
        3: "Eb",
        4: "E",
        5: "F",
        6: "F#",
        7: "G",
        8: "Ab",
        9: "A",
        10: "Bb",
        11: "B",
    }
    for root in range(12):
        major = sorted([(root + interval) % 12 for interval in major_template])
        minor = sorted([(root + interval) % 12 for interval in minor_template])
        if mods == major:
            return note_names[root]
        if mods == minor:
            return note_names[root] + "m"
    return "Unknown"


# *****************************************************
#
#   convert_csv_to_midi
#
#   Brief: Logic for converting a .csv file to a .mid file
#
# *****************************************************
def convert_csv_to_midi(csv_file, output_folder="midi_files"):
    """
    Converts a chord progression CSV file to a MIDI file.
    Each chord is held from its timestamp until the next chord (or for 1 second for the last chord).
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    df = pd.read_csv(csv_file)
    df = df.sort_values("Timestamp")
    ticks_per_beat = 480
    ticks_per_second = 960  # assuming 120 BPM: 1 second = 960 ticks
    mid = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=500000, time=0))
    events = []
    timestamps = df["Timestamp"].tolist()
    chords = df["Chord"].tolist()
    for i in range(len(timestamps)):
        start_tick = int(timestamps[i] * ticks_per_second)
        if i < len(timestamps) - 1:
            end_tick = int(timestamps[i + 1] * ticks_per_second)
        else:
            end_tick = start_tick + ticks_per_second  # last chord duration: 1 second
        notes = chord_to_midi_notes(chords[i])
        events.append((start_tick, "note_on", notes))
        events.append((end_tick, "note_off", notes))
    events.sort(key=lambda x: x[0])
    prev_tick = 0
    for event in events:
        tick, event_type, notes = event
        delta = tick - prev_tick
        for note in notes:
            track.append(mido.Message(event_type, note=note, velocity=64, time=delta))
            delta = 0
        prev_tick = tick
    output_midi = os.path.join(
        output_folder, os.path.basename(csv_file).replace(".csv", ".mid")
    )
    mid.save(output_midi)
    print(f"MIDI file saved to: {output_midi}")


# *****************************************************
#
#   process_csv_folder
#
#   Brief: Convert a folder of .csv files to a folder of .mid files
#
# *****************************************************
def process_csv_folder(input_folder, output_folder="midi_files"):
    """
    Converts all CSV files in the input folder to MIDI files.
    """
    if not os.path.exists(input_folder):
        print(f"Input folder '{input_folder}' does not exist.")
        return
    csv_files = [f for f in os.listdir(input_folder) if f.lower().endswith(".csv")]
    if not csv_files:
        print(f"No CSV files found in '{input_folder}'.")
        return
    print(
        f"Found {len(csv_files)} CSV file(s) in '{input_folder}'. Converting to MIDI..."
    )
    for csv_file in csv_files:
        full_path = os.path.join(input_folder, csv_file)
        convert_csv_to_midi(full_path, output_folder)
    print("CSV to MIDI conversion complete.")


# *****************************************************
#
#   convert_midi_to_text
#
#   Brief: Logic for converting a .midi file to a .txt file
#
# *****************************************************
def convert_midi_to_text(midi_file, output_folder="midi_texts"):
    """
    Converts a MIDI file to a plain text representation.
    The output text file lists each track and, for each track, every MIDI event with its absolute tick time.
    This text file format is designed to be easy to read and parse.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    mid = mido.MidiFile(midi_file)
    lines = []
    for i, track in enumerate(mid.tracks):
        lines.append(f"Track {i}: {track.name if track.name else 'Unnamed'}")
        abs_tick = 0
        for msg in track:
            abs_tick += msg.time
            # Represent the message in a simplified text format.
            lines.append(f"Time: {abs_tick} | {msg}")
        lines.append("")  # Blank line between tracks
    base_name = os.path.splitext(os.path.basename(midi_file))[0]
    output_file = os.path.join(output_folder, f"{base_name}.txt")
    with open(output_file, "w") as f:
        for line in lines:
            f.write(line + "\n")
    print(f"Text file saved to: {output_file}")


# *****************************************************
#
#   convert_midi_to_json
#
#   Brief: Logic for converting a .midi file to a .json file
#
# *****************************************************
def convert_midi_to_json(midi_file, json_file):
    """
    Convert a MIDI file to JSON format using the mido library.

    Args:
        midi_file (str): Path to the input MIDI file.
        json_file (str): Path where the output JSON file will be saved.
    """
    try:
        # Read the MIDI file
        midi = mido.MidiFile(midi_file)

        # Initialize a list to hold all tracks data
        tracks_data = []

        # Iterate through each track in the MIDI file
        for i, track in enumerate(midi.tracks):
            track_messages = []
            # Extract all messages from the track
            for msg in track:
                # Convert each message to a dictionary (JSON-serializable)
                track_messages.append(msg.dict())
            # Store track index and its messages
            tracks_data.append({"track": i, "messages": track_messages})

        # Create a dictionary to hold all MIDI data
        midi_data = {"tracks": tracks_data}

        # Write the MIDI data to a JSON file
        with open(json_file, "w") as f:
            json.dump(midi_data, f, indent=2)

        print(f"Successfully converted {midi_file} to {json_file}")

    except Exception as e:
        print(f"Error converting {midi_file}: {str(e)}")


# *****************************************************
#
#   process_midi_folder_json_conversion
#
#   Brief: Used for converting a folder containing .mid files to a folder containing .json files
#
# *****************************************************
def process_midi_folder_json_conversion(input_folder, output_folder):
    """
    Converts all MIDI files in the input folder to JSON files.
    """
    if not os.path.exists(input_folder):
        print(f"Input folder '{input_folder}' does not exist")
        return

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    midi_files = [
        f for f in os.listdir(input_folder) if f.lower().endswith((".mid", ".midi"))
    ]
    if not midi_files:
        print(f"No MIDI files found in '{input_folder}'")
        return

    for midi_file in midi_files:
        input_file = os.path.join(input_folder, midi_file)
        base_name = os.path.basename(midi_file)
        name, _ = os.path.splitext(base_name)
        json_file = os.path.join(output_folder, name + ".json")
        convert_midi_to_json(input_file, json_file)


# *****************************************************
#
#   NEW FUNCTION: convert_midi_to_csv (Chord Extraction)
#
#   Brief: Converts a MIDI file to a CSV file containing chord progression data.
#
#   CSV format:
#   Timestamp,Chord
#   0.53,Ab
#   1.09,Ab
#   1.61,Ab
#   ...
#
# *****************************************************
def convert_midi_to_csv(midi_file, output_folder="midi_csv"):
    """
    Converts a MIDI file to CSV format with chord progression information.
    Assumes the MIDI file contains chord events as groups of note_on messages in track 0.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    mid = mido.MidiFile(midi_file)
    ticks_per_beat = mid.ticks_per_beat
    # Get tempo from first set_tempo event; default to 500000 microseconds per beat.
    tempo = 500000
    for track in mid.tracks:
        for msg in track:
            if msg.type == "set_tempo":
                tempo = msg.tempo
                break
        if tempo != 500000:
            break
    seconds_per_tick = tempo / (ticks_per_beat * 1e6)

    # Process track 0 for chord events
    abs_tick = 0
    chord_events = []  # list of tuples (abs_tick, [note_numbers])
    current_group = None
    for msg in mid.tracks[0]:
        abs_tick += msg.time
        if msg.type == "note_on" and msg.velocity > 0:
            # Group note_on events that occur at the same absolute tick
            if current_group is not None and current_group[0] == abs_tick:
                current_group[1].append(msg.note)
            else:
                current_group = (abs_tick, [msg.note])
                chord_events.append(current_group)
    # For each chord event, detect chord label
    output_data = []
    for tick, notes in chord_events:
        timestamp = tick * seconds_per_tick
        chord_label = detect_chord(notes)
        output_data.append({"Timestamp": round(timestamp, 2), "Chord": chord_label})

    # Create DataFrame and save CSV
    df = pd.DataFrame(output_data)
    base_name = os.path.basename(midi_file)
    name, _ = os.path.splitext(base_name)
    output_csv = os.path.join(output_folder, name + ".csv")
    df.to_csv(output_csv, index=False)
    print(f"Chord CSV file saved to: {output_csv}")
    return df


# *****************************************************
#
#   get_spotify_credentials()
#
#   Brief:
#
# *****************************************************
def get_spotify_credentials(credentials_file: str):
    """
    Reads Spotify API credentials from a JSON file.
    """
    with open(credentials_file, "r") as file:
        credentials = json.load(file)
    return credentials["client_id"], credentials["client_secret"]


# *****************************************************
#
#   generate_input_csv()
#
#   Brief:
#
# *****************************************************
def generate_input_csv(
    playlist_url: str, output_csv: str, client_id: str, client_secret: str
):
    """
    Fetches song IDs and names from a Spotify playlist and saves them to a CSV file.
    """
    sp = spotipy.Spotify(
        auth_manager=SpotifyOAuth(
            client_id=client_id,
            client_secret=client_secret,
            redirect_uri="http://localhost:8080",
            scope="playlist-read-private",
        )
    )

    # Extract playlist ID from URL
    playlist_id = playlist_url.split("playlist/")[1].split("?")[0]

    results = sp.playlist_tracks(playlist_id)

    song_data = []
    for track in results["items"]:
        song_id = track["track"]["id"]
        song_name = track["track"]["name"]
        song_data.append([song_id, song_name])

    df = pd.DataFrame(song_data, columns=["id", "name"])
    df.to_csv(output_csv, index=False)
    print(f"CSV file saved: {output_csv}")


# *****************************************************
#
#   process_input_metadata()
#
#   Brief:
#
# *****************************************************
def process_input_metadata(input_csv: str, client_id: str, client_secret: str):
    """
    Reads the CSV file and fetches detailed song attributes from Spotify API.
    """
    sp = spotipy.Spotify(
        auth_manager=SpotifyOAuth(
            client_id=client_id,
            client_secret=client_secret,
            redirect_uri="http://localhost:8080",
            scope="user-library-read",
        )
    )

    df = pd.read_csv(input_csv)

    metadata_list = []
    for song_id in df["id"]:
        try:
            time.sleep(0.5)  # Avoid rate limits
            attributes = sp.audio_features([song_id])  # Pass as a list
            if attributes and attributes[0]:
                metadata_list.append(json.dumps(attributes[0]))
            else:
                metadata_list.append("{}")  # Store empty JSON if data is missing
        except spotipy.SpotifyException as e:
            print(f"Error fetching audio features for {song_id}: {e}")
            metadata_list.append("{}")  # Store empty JSON on error

    df["attributes"] = metadata_list
    df.to_csv(input_csv, index=False)
    print(f"Updated CSV file with metadata: {input_csv}")


# *****************************************************
#
#   create_database_pipeline()
#
#   Brief: The following function contains all the logic to execute the database creation pipeline
#
# *****************************************************
def databasegen(url, credentials_file) -> bool:
    """
    Main function to generate the training dataset by processing input metadata,
    extracting song attributes, analyzing audio, and merging input-output pairs.

    :param url: The URL containing the playlist or song list to process.
    :return: Boolean indicating success or failure.
    """

    """
    As this function is developed uncomment the code function calls below
    
    """

    # Retrieve API credentials
    client_id, client_secret = get_spotify_credentials(credentials_file)

    # Step 1: Generate Input Metadata (.csv)
    input_csv_path = "playlist_data.csv"
    generate_input_csv(url, input_csv_path, client_id, client_secret)

    # # Step 2: Process Input Metadata
    # process_input_metadata(input_csv_path, client_id, client_secret)

    # paused dev of Input pipeline because of deprecated audio-features enpoint

    # # Step 3: Download and Convert Audio
    # # Use spotdl to download .mp3 files, convert them to .wav, and prepare for analysis.
    # wav_directory = process_audio_files(input_csv_path)

    # # Step 4: Analyze Audio Data
    # # Extract harmonic, beat, and chord data, storing results in CSV format.
    # analyzed_data_dir = analyze_audio(wav_directory)

    # # Step 5: Convert Processed Data to JSON
    # # Convert the analyzed data (stored in CSV format) into JSON for training.
    # analyzed_json_dir = convert_analyzed_data_to_json(analyzed_data_dir)

    # # Step 6: Merge Input and Output Data
    # # Combine input metadata with the corresponding analyzed song data.
    # final_dataset_path = merge_input_output(input_csv_path, analyzed_json_dir)

    # Step 7: Output the location of the final dataset
    # print(f"Dataset successfully created: {final_dataset_path}")
    print("Output: /final/dataset/path/file.json")

    return False  # Adjust as needed for error handling


### MAIN FUNCTION WITH COMMAND-LINE INTERFACE ###
def main():
    parser = argparse.ArgumentParser(
        description="Music Tool: Convert MP3s to WAVs, extract chord progressions, visualize chords, convert CSV to MIDI, convert MIDI to text, convert MIDI to JSON, or convert MIDI to CSV."
    )
    subparsers = parser.add_subparsers(dest="command", help="Sub-command help")

    # Subcommand: convert
    parser_convert = subparsers.add_parser(
        "convert", help="Convert MP3 files in a folder to WAV format."
    )
    parser_convert.add_argument(
        "--input_folder",
        type=str,
        default="unconverted_mp3s",
        help="Folder containing MP3 files to convert.",
    )
    parser_convert.add_argument(
        "--output_folder",
        type=str,
        default="converted_wavs",
        help="Folder to save converted WAV files.",
    )

    # Subcommand: extract
    parser_extract = subparsers.add_parser(
        "extract", help="Extract chord progressions from WAV files in a folder."
    )
    parser_extract.add_argument(
        "--input_folder",
        type=str,
        default="converted_wavs",
        help="Folder containing WAV files.",
    )
    parser_extract.add_argument(
        "--output_folder",
        type=str,
        default="chord_data",
        help="Folder to save chord CSV files.",
    )

    # Subcommand: visualize
    parser_visualize = subparsers.add_parser(
        "visualize", help="Visualize chord progression from a CSV file."
    )
    parser_visualize.add_argument(
        "--csv_file",
        type=str,
        required=True,
        help="CSV file containing chord progression data.",
    )
    parser_visualize.add_argument(
        "--save_folder",
        type=str,
        default="chord_visualizations",
        help="Folder to save the chord visualization image.",
    )
    parser_visualize.add_argument(
        "--no_show", action="store_true", help="Do not display the plot, only save it."
    )

    # Subcommand: csv2midi
    parser_csv2midi = subparsers.add_parser(
        "csv2midi", help="Convert CSV files to MIDI files."
    )
    parser_csv2midi.add_argument(
        "--csv_file", type=str, help="Single CSV file to convert to MIDI."
    )
    parser_csv2midi.add_argument(
        "--input_folder", type=str, help="Folder containing CSV files to convert."
    )
    parser_csv2midi.add_argument(
        "--output_folder",
        type=str,
        default="midi_files",
        help="Folder to save MIDI files.",
    )

    # Subcommand: midi2txt
    parser_midi2txt = subparsers.add_parser(
        "midi2txt", help="Convert MIDI files to text files."
    )
    parser_midi2txt.add_argument(
        "--midi_file", type=str, help="Single MIDI file to convert to text."
    )
    parser_midi2txt.add_argument(
        "--input_folder", type=str, help="Folder containing MIDI files to convert."
    )
    parser_midi2txt.add_argument(
        "--output_folder",
        type=str,
        default="midi_texts",
        help="Folder to save text files.",
    )

    # Subcommand: midi2json
    parser_midi2json = subparsers.add_parser(
        "midi2json", help="Convert MIDI files to JSON files."
    )
    parser_midi2json.add_argument(
        "--midi_file", type=str, help="Single MIDI file to convert to JSON."
    )
    parser_midi2json.add_argument(
        "--input_folder", type=str, help="Folder containing MIDI files to convert."
    )
    parser_midi2json.add_argument(
        "--output_folder",
        type=str,
        default="converted_jsons",
        help="Folder to save JSON files (default: 'converted_jsons').",
    )

    # NEW Subcommand: midi2csv
    parser_midi2csv = subparsers.add_parser(
        "midi2csv", help="Convert MIDI file(s) to CSV files (chord progression)."
    )
    parser_midi2csv.add_argument(
        "--midi_file", type=str, help="Single MIDI file to convert to CSV."
    )
    parser_midi2csv.add_argument(
        "--input_folder", type=str, help="Folder containing MIDI files to convert."
    )
    parser_midi2csv.add_argument(
        "--output_folder",
        type=str,
        default="midi_csv",
        help="Folder to save CSV files.",
    )

    # Subcommand: databasegen
    parser_databasegen = subparsers.add_parser(
        "databasegen", help="Initiate database gen. pipeline."
    )
    parser_databasegen.add_argument(
        "--input_url", type=str, help="Provide Spotify playlist URL"
    )
    parser_databasegen.add_argument(
        "--credentials_path",
        type=str,
        help="Provide Spotify Client ID & Secret in .json",
    )

    args = parser.parse_args()

    if args.command == "convert":
        convert_folder(args.input_folder, args.output_folder)
    elif args.command == "extract":
        process_wav_folder(args.input_folder, args.output_folder)
    elif args.command == "visualize":
        visualize_chords_on_y_axis(
            args.csv_file, args.save_folder, show_plot=not args.no_show
        )
    elif args.command == "csv2midi":
        if args.csv_file:
            convert_csv_to_midi(args.csv_file, args.output_folder)
        elif args.input_folder:
            process_csv_folder(args.input_folder, args.output_folder)
        else:
            print(
                "Please specify either --csv_file for a single file or --input_folder for a folder of CSV files."
            )
    elif args.command == "midi2txt":
        if args.midi_file:
            convert_midi_to_text(args.midi_file, args.output_folder)
        elif args.input_folder:
            process_midi_folder(args.input_folder, args.output_folder)
        else:
            print(
                "Please specify either --midi_file for a single file or --input_folder for a folder of MIDI files."
            )
    elif args.command == "midi2json":
        if args.midi_file:
            if not os.path.exists(args.midi_file):
                print(f"MIDI file '{args.midi_file}' does not exist")
                return
            if not args.midi_file.lower().endswith((".mid", ".midi")):
                print(f"'{args.midi_file}' is not a MIDI file")
                return

            if not os.path.exists(args.output_folder):
                os.makedirs(args.output_folder)

            base_name = os.path.basename(args.midi_file)
            name, _ = os.path.splitext(base_name)
            json_file = os.path.join(args.output_folder, name + ".json")
            convert_midi_to_json(args.midi_file, json_file)
        elif args.input_folder:
            process_midi_folder_json_conversion(args.input_folder, args.output_folder)
        else:
            print(
                "Please specify either --midi_file for a single file or --input_folder for a folder of MIDI files."
            )
    elif args.command == "midi2csv":
        if args.midi_file:
            convert_midi_to_csv(args.midi_file, args.output_folder)
        elif args.input_folder:
            midi_files = [
                f
                for f in os.listdir(args.input_folder)
                if f.lower().endswith((".mid", ".midi"))
            ]
            if not midi_files:
                print(f"No MIDI files found in '{args.input_folder}'.")
                return
            for midi_file in midi_files:
                full_path = os.path.join(args.input_folder, midi_file)
                convert_midi_to_csv(full_path, args.output_folder)
        else:
            print(
                "Please specify either --midi_file for a single file or --input_folder for a folder of MIDI files."
            )
    elif args.command == "databasegen":
        if args.input_url:
            databasegen(args.input_url, args.credentials_path)
        else:
            print("Please specify --input_url a valid spotify playlist url")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
