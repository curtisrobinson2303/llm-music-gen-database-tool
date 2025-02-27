import os
import argparse
import mido
import json

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
        with open(json_file, 'w') as f:
            json.dump(midi_data, f, indent=2)
        
        print(f"Successfully converted {midi_file} to {json_file}")
    
    except Exception as e:
        print(f"Error converting {midi_file}: {str(e)}")

def process_midi_folder(input_folder, output_folder):
    """
    Process all MIDI files in a folder and convert them to JSON.
    
    Args:
        input_folder (str): Folder containing MIDI files.
        output_folder (str): Folder to save JSON files.
    """
    if not os.path.exists(input_folder):
        print(f"Input folder '{input_folder}' does not exist")
        return
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    midi_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.mid', '.midi'))]
    if not midi_files:
        print(f"No MIDI files found in '{input_folder}'")
        return
    
    for midi_file in midi_files:
        input_file = os.path.join(input_folder, midi_file)
        base_name = os.path.basename(midi_file)
        name, _ = os.path.splitext(base_name)
        json_file = os.path.join(output_folder, name + '.json')
        convert_midi_to_json(input_file, json_file)

def main():
    """
    Main function with command-line interface using argparse.
    """
    parser = argparse.ArgumentParser(
        description="MIDI Tool: Convert MIDI files to JSON format."
    )
    subparsers = parser.add_subparsers(dest="command", help="Sub-command help")

    # Subcommand: midi2json
    parser_midi2json = subparsers.add_parser("midi2json", help="Convert MIDI files to JSON files.")
    parser_midi2json.add_argument("--midi_file", type=str, help="Single MIDI file to convert to JSON.")
    parser_midi2json.add_argument("--input_folder", type=str, help="Folder containing MIDI files to convert.")
    parser_midi2json.add_argument("--output_folder", type=str, default="converted_jsons", 
                                 help="Folder to save JSON files (default: 'converted_jsons').")

    args = parser.parse_args()

    if args.command == "midi2json":
        if args.midi_file:
            # Convert a single MIDI file
            if not os.path.exists(args.midi_file):
                print(f"MIDI file '{args.midi_file}' does not exist")
                return
            if not args.midi_file.lower().endswith(('.mid', '.midi')):
                print(f"'{args.midi_file}' is not a MIDI file")
                return
            
            if not os.path.exists(args.output_folder):
                os.makedirs(args.output_folder)
            
            base_name = os.path.basename(args.midi_file)
            name, _ = os.path.splitext(base_name)
            json_file = os.path.join(args.output_folder, name + '.json')
            convert_midi_to_json(args.midi_file, json_file)
        
        elif args.input_folder:
            # Convert all MIDI files in a folder
            process_midi_folder(args.input_folder, args.output_folder)
        
        else:
            print("Please specify either --midi_file for a single file or --input_folder for a folder of MIDI files.")
            parser_midi2json.print_help()
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()