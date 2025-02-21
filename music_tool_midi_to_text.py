#!/usr/bin/env python
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import essentia.standard as es
from pydub import AudioSegment
import mido
from mido import Message, MidiFile, MidiTrack

### AUDIO CONVERSION FUNCTIONS ###
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

def convert_folder(input_folder, output_folder="converted_wavs"):
    """
    Converts all MP3 files in the given input folder to WAV format.
    """
    if not os.path.exists(input_folder):
        print(f"Input folder '{input_folder}' does not exist.")
        return
    files = os.listdir(input_folder)
    mp3_files = [f for f in files if f.lower().endswith('.mp3')]
    if not mp3_files:
        print("No MP3 files found in the input folder.")
        return
    print(f"Found {len(mp3_files)} MP3 file(s) in '{input_folder}'. Starting conversion...")
    for file in mp3_files:
        input_file_path = os.path.join(input_folder, file)
        convert_mp3_to_wav(input_file_path, output_folder)
    print("Conversion complete.")

### CHORD EXTRACTION FUNCTIONS ###
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
    frame_generator = es.FrameGenerator(audio, frameSize=frame_size, hopSize=hop_size, startFromZero=True)
    windowing = es.Windowing(type='hann')
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
    output_csv = os.path.join(output_folder, os.path.basename(wav_file).replace(".wav", "_chords.csv"))
    df.to_csv(output_csv, index=False)
    print(f"Chords extracted and saved to: {output_csv}\n")
    return df

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

### CHORD VISUALIZATION FUNCTION ###
def visualize_chords_on_y_axis(csv_file, save_folder="chord_visualizations", show_plot=True):
    """
    Visualizes the chord progression with time on the x-axis and chords on the y-axis.
    """
    df = pd.read_csv(csv_file)
    unique_chords = sorted(df['Chord'].unique())
    chord_to_num = {chord: i for i, chord in enumerate(unique_chords)}
    df['ChordNum'] = df['Chord'].map(chord_to_num)
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.scatter(df['Timestamp'], df['ChordNum'], color='blue', s=50)
    ax.set_yticks(list(chord_to_num.values()))
    ax.set_yticklabels(list(chord_to_num.keys()))
    ax.set_xlabel("Time (seconds)")
    ax.set_title("Chord Progression Over Time")
    ax.grid(axis='x', linestyle='--', alpha=0.7)
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

### CSV TO MIDI CONVERSION FUNCTIONS ###
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
        "C": 60, "C#": 61, "Db": 61,
        "D": 62, "D#": 63, "Eb": 63,
        "E": 64,
        "F": 65, "F#": 66, "Gb": 66,
        "G": 67, "G#": 68, "Ab": 68,
        "A": 69, "A#": 70, "Bb": 70,
        "B": 71
    }
    if root in note_to_midi:
        base = note_to_midi[root]
    else:
        base = 60  # default to C if unknown
    return [base + interval for interval in intervals]

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
    mid = MidiFile(ticks_per_beat=ticks_per_beat)
    track = MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage('set_tempo', tempo=500000, time=0))
    events = []
    timestamps = df['Timestamp'].tolist()
    chords = df['Chord'].tolist()
    for i in range(len(timestamps)):
        start_tick = int(timestamps[i] * ticks_per_second)
        if i < len(timestamps) - 1:
            end_tick = int(timestamps[i+1] * ticks_per_second)
        else:
            end_tick = start_tick + ticks_per_second  # last chord duration: 1 second
        notes = chord_to_midi_notes(chords[i])
        events.append((start_tick, 'note_on', notes))
        events.append((end_tick, 'note_off', notes))
    events.sort(key=lambda x: x[0])
    prev_tick = 0
    for event in events:
        tick, event_type, notes = event
        delta = tick - prev_tick
        for note in notes:
            track.append(Message(event_type, note=note, velocity=64, time=delta))
            delta = 0
        prev_tick = tick
    output_midi = os.path.join(output_folder, os.path.basename(csv_file).replace(".csv", ".mid"))
    mid.save(output_midi)
    print(f"MIDI file saved to: {output_midi}")

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
    print(f"Found {len(csv_files)} CSV file(s) in '{input_folder}'. Converting to MIDI...")
    for csv_file in csv_files:
        full_path = os.path.join(input_folder, csv_file)
        convert_csv_to_midi(full_path, output_folder)
    print("CSV to MIDI conversion complete.")

### MIDI TO TEXT CONVERSION FUNCTIONS ###
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

def process_midi_folder(input_folder, output_folder="midi_texts"):
    """
    Converts all MIDI files in the input folder to text files.
    """
    if not os.path.exists(input_folder):
        print(f"Input folder '{input_folder}' does not exist.")
        return
    midi_files = [f for f in os.listdir(input_folder) if f.lower().endswith(".mid")]
    if not midi_files:
        print(f"No MIDI files found in '{input_folder}'.")
        return
    print(f"Found {len(midi_files)} MIDI file(s) in '{input_folder}'. Converting to text...")
    for midi_file in midi_files:
        full_path = os.path.join(input_folder, midi_file)
        convert_midi_to_text(full_path, output_folder)
    print("MIDI to text conversion complete.")

### MAIN FUNCTION WITH COMMAND-LINE INTERFACE ###
def main():
    parser = argparse.ArgumentParser(
        description="Music Tool: Convert MP3s to WAVs, extract chord progressions, visualize chords, convert CSV to MIDI, or convert MIDI to text."
    )
    subparsers = parser.add_subparsers(dest="command", help="Sub-command help")

    # Subcommand: convert
    parser_convert = subparsers.add_parser("convert", help="Convert MP3 files in a folder to WAV format.")
    parser_convert.add_argument("--input_folder", type=str, default="unconverted_mp3s", help="Folder containing MP3 files to convert.")
    parser_convert.add_argument("--output_folder", type=str, default="converted_wavs", help="Folder to save converted WAV files.")

    # Subcommand: extract
    parser_extract = subparsers.add_parser("extract", help="Extract chord progressions from WAV files in a folder.")
    parser_extract.add_argument("--input_folder", type=str, default="converted_wavs", help="Folder containing WAV files.")
    parser_extract.add_argument("--output_folder", type=str, default="chord_data", help="Folder to save chord CSV files.")

    # Subcommand: visualize
    parser_visualize = subparsers.add_parser("visualize", help="Visualize chord progression from a CSV file.")
    parser_visualize.add_argument("--csv_file", type=str, required=True, help="CSV file containing chord progression data.")
    parser_visualize.add_argument("--save_folder", type=str, default="chord_visualizations", help="Folder to save the chord visualization image.")
    parser_visualize.add_argument("--no_show", action="store_true", help="Do not display the plot, only save it.")

    # Subcommand: csv2midi
    parser_csv2midi = subparsers.add_parser("csv2midi", help="Convert CSV files to MIDI files.")
    parser_csv2midi.add_argument("--csv_file", type=str, help="Single CSV file to convert to MIDI.")
    parser_csv2midi.add_argument("--input_folder", type=str, help="Folder containing CSV files to convert.")
    parser_csv2midi.add_argument("--output_folder", type=str, default="midi_files", help="Folder to save MIDI files.")

    # Subcommand: midi2txt
    parser_midi2txt = subparsers.add_parser("midi2txt", help="Convert MIDI files to text files.")
    parser_midi2txt.add_argument("--midi_file", type=str, help="Single MIDI file to convert to text.")
    parser_midi2txt.add_argument("--input_folder", type=str, help="Folder containing MIDI files to convert.")
    parser_midi2txt.add_argument("--output_folder", type=str, default="midi_texts", help="Folder to save text files.")

    args = parser.parse_args()

    if args.command == "convert":
        convert_folder(args.input_folder, args.output_folder)
    elif args.command == "extract":
        process_wav_folder(args.input_folder, args.output_folder)
    elif args.command == "visualize":
        visualize_chords_on_y_axis(args.csv_file, args.save_folder, show_plot=not args.no_show)
    elif args.command == "csv2midi":
        if args.csv_file:
            convert_csv_to_midi(args.csv_file, args.output_folder)
        elif args.input_folder:
            process_csv_folder(args.input_folder, args.output_folder)
        else:
            print("Please specify either --csv_file for a single file or --input_folder for a folder of CSV files.")
    elif args.command == "midi2txt":
        if args.midi_file:
            convert_midi_to_text(args.midi_file, args.output_folder)
        elif args.input_folder:
            process_midi_folder(args.input_folder, args.output_folder)
        else:
            print("Please specify either --midi_file for a single file or --input_folder for a folder of MIDI files.")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
