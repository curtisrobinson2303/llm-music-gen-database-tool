import os
import json


def create_input_output_pairs(input_pair_dir, output_pair_dir, output_file_path):
    if not os.path.isdir(input_pair_dir):
        raise FileNotFoundError(f"Input directory '{input_pair_dir}' does not exist.")
    if not os.path.isdir(output_pair_dir):
        raise FileNotFoundError(f"Output directory '{output_pair_dir}' does not exist.")

    if not output_file_path.endswith(".json"):
        output_file_path += ".json"

    input_files = {
        os.path.splitext(f)[0]: f
        for f in os.listdir(input_pair_dir)
        if f.endswith(".json")
    }
    output_files_raw = [f for f in os.listdir(output_pair_dir) if f.endswith(".json")]

    # Normalize output filenames: remove '_chords' suffix before comparing
    output_files = {}
    for f in output_files_raw:
        base_name = os.path.splitext(f)[0]
        if base_name.endswith("_chords"):
            base_name = base_name[:-7]  # remove "_chords"
        output_files[base_name] = f

    print(f"Input files (base names): {input_files.keys()}")
    print(f"Output files (base names): {output_files.keys()}")

    common_keys = set(input_files.keys()) & set(output_files.keys())
    print(f"Common base filenames: {common_keys}")

    if not common_keys:
        raise ValueError(
            "No matching JSON files found between input and output directories."
        )

    combined_data = {}

    for key in sorted(common_keys):
        input_path = os.path.join(input_pair_dir, input_files[key])
        output_path = os.path.join(output_pair_dir, output_files[key])

        with open(input_path, "r", encoding="utf-8") as f:
            input_json = json.load(f)

        with open(output_path, "r", encoding="utf-8") as f:
            output_json = json.load(f)

        combined_data[key] = {
            "input_json": input_json,
            "output_json": output_json,
        }

    with open(output_file_path, "w", encoding="utf-8") as f:
        json.dump(combined_data, f, indent=2, ensure_ascii=False)

    absolute_output_path = os.path.abspath(output_file_path)
    print(
        f"✅ Successfully created {absolute_output_path} with {len(combined_data)} pairs."
    )

    return absolute_output_path


# --- Run it ---
print("\n\n\nCOMBINING INPUT & OUTPUT JSON DATA\n\n\n")
create_input_output_pairs(
    "input_pair_json_data", "output_pair_json_data", "input_output_dataset.json"
)
