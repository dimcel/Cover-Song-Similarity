import os
import h5py
import argparse
from collections import Counter

def check_hpcp_lengths(target_folder: str):
    hpcp_frame_lengths = []

    # Ensure the target folder exists
    if not os.path.exists(target_folder):
        print(f"Error: The folder '{target_folder}' does not exist.")
        return

    # Process each folder inside the target directory
    for wid_folder in os.listdir(target_folder):
        wid_path = os.path.join(target_folder, wid_folder)
        
        if os.path.isdir(wid_path):
            for file in os.listdir(wid_path):
                if file.endswith(".h5"):
                    file_path = os.path.join(wid_path, file)

                    try:
                        with h5py.File(file_path, "r") as h5f:
                            if "hpcp" in h5f:
                                hpcp_data = h5f["hpcp"][:]

                                # Get both dimensions of the HPCP matrix
                                num_frames = hpcp_data.shape[0]
                                num_pitch_classes = hpcp_data.shape[1]

                                hpcp_frame_lengths.append(num_frames)

                                if num_pitch_classes != 12:
                                    print(f"Warning: Unexpected number of pitch classes in {file}: {num_pitch_classes}")
                    
                    except Exception as e:
                        print(f"Error reading {file}: {e}")

    # Analyze results
    unique_lengths = set(hpcp_frame_lengths)

    print(f"\nUnique HPCP frame lengths found: {unique_lengths}")

    if len(unique_lengths) == 1:
        print(f"All files have the same HPCP frame length: {unique_lengths.pop()}")
    else:
        print("Files have different HPCP frame lengths!")
        
        # Count occurrences of each HPCP length
        length_counts = Counter(hpcp_frame_lengths)

        print("\nHPCP Frame Length Distribution:")
        for length, count in sorted(length_counts.items()):
            print(f"  - {count} files have HPCP frame length {length}")

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Check HPCP frame lengths in HDF5 files.")
    parser.add_argument('--target_folder', type=str, required=True, help="Path to the target folder to check.")

    args = parser.parse_args()

    # Run the check with the provided folder
    check_hpcp_lengths(args.target_folder)

if __name__ == "__main__":
    main()

# python length_check.py --target_folder "../data/17K_data"