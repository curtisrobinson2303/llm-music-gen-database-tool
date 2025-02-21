#!/usr/bin/env python
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import essentia.standard as es
from pydub import AudioSegment

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
    The resulting WAV files are saved in the specified output folder.
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
    The CSV is saved in the specified output folder.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Load audio
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

    # Compute HPCP features for each frame (as list-of-lists of floats)
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

    # Align chords with beat timestamps and create DataFrame
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
    The CSV output for each file is saved in the output folder.
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
    Reads a CSV file with 'Timestamp' and 'Chord' columns, maps chords to a numeric y-axis,
    and saves the plot in the specified folder.
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

### MAIN FUNCTION WITH COMMAND-LINE INTERFACE ###
def main():
    parser = argparse.ArgumentParser(
        description="Music Tool: Convert MP3s to WAVs, extract chord progressions, or visualize chords."
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

    args = parser.parse_args()

    if args.command == "convert":
        convert_folder(args.input_folder, args.output_folder)
    elif args.command == "extract":
        process_wav_folder(args.input_folder, args.output_folder)
    elif args.command == "visualize":
        visualize_chords_on_y_axis(args.csv_file, args.save_folder, show_plot=not args.no_show)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
