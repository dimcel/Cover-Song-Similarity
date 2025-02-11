import os
import h5py
import numpy as np
import argparse
from tqdm import tqdm

def process_files(source_dir: str, target_dir: str, target_length: int):
    os.makedirs(target_dir, exist_ok=True)

    work_folders = [f for f in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, f))]

    with tqdm(total=len(work_folders), desc="Processing Folders", unit="folder") as pbar:
        for work_folder in work_folders:
            work_path = os.path.join(source_dir, work_folder)
            new_work_path = os.path.join(target_dir, work_folder)

            os.makedirs(new_work_path, exist_ok=True)
            
            # Get all .h5 files in the folder
            h5_files = [f for f in os.listdir(work_path) if f.endswith(".h5")]

            for file in tqdm(h5_files, desc=f"Processing {work_folder}", leave=False, unit="file"):
                old_file_path = os.path.join(work_path, file)
                new_file_path = os.path.join(new_work_path, file)
                
                try:
                    with h5py.File(old_file_path, "r") as h5f:
                        if "hpcp" in h5f:
                            hpcp_data = h5f["hpcp"][:]
                            
                            current_length = hpcp_data.shape[0]
                            
                            if current_length > target_length:
                                hpcp_data = hpcp_data[:target_length, :]  # cut
                            elif current_length < target_length:
                                padding = np.zeros((target_length - current_length, hpcp_data.shape[1]))
                                hpcp_data = np.vstack((hpcp_data, padding))  # pad

                    with h5py.File(new_file_path, "w") as new_h5f:
                        new_h5f.create_dataset("hpcp", data=hpcp_data)

                except Exception as e:
                    print(f"\nError processing {file}: {e}")

            pbar.update(1)

def main():
    parser = argparse.ArgumentParser(description="Process HPCP files and adjust their length.")
    parser.add_argument('--source_folder', type=str, required=True, help="Path to the source folder containing HPCP files.")
    parser.add_argument('--output_folder', type=str, required=True, help="Path to the target folder to save processed files.")
    parser.add_argument('--target_length', type=int, required=True, help="The target length to resize the HPCP matrices.")

    args = parser.parse_args()

    process_files(args.source_folder, args.output_folder, args.target_length)

if __name__ == "__main__":
    main()

# python fix_length.py --source_folder "../data/da-tacos_benchmark_subset_hpcp" --output_folder "../data/17K_data" --target_length 17000  