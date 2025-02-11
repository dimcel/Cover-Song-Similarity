import os
import random
import json
import argparse
from typing import List, Tuple, Dict

def get_all_folders(parent_folder: str) -> Dict[str, List[str]]:
    """
    Reads all subfolders in the parent directory and stores filenames.
    
    :param parent_folder: Path to the main dataset directory
    :return: Dictionary {folder_name: [list of performances]}
    """
    data = {}
    for folder in os.listdir(parent_folder):
        folder_path = os.path.join(parent_folder, folder)
        if os.path.isdir(folder_path):
            performances = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
            if performances:  # Only store folders that contain files
                data[folder] = performances
    return data


def create_song_pairs(data: Dict[str, List[str]], num_pairs: int) -> List[Dict[str, List[str] | int]]:
    """
    Creates labeled pairs of songs from work folders in the correct format.

    :param data: Dictionary {work_folder: [list of performances]}
    :param num_pairs: Number of similar and dissimilar pairs to generate (each)
    :return: A list of dictionaries [{"Pair": [song1, song2], "Label": label}, ...]
    """

    song_pairs = []
    all_folders = list(data.keys())

    # Generate Similar Pairs (Label 0)
    while len(song_pairs) < num_pairs:
        folder = random.choice(all_folders)
        if len(data[folder]) < 2:
            continue 
        song1, song2 = random.sample(data[folder], 2) #ensres two different songs
        song_pairs.append({
            "Pair": [f"{folder}/{song1}", f"{folder}/{song2}"],
            "Label": 0
        })

    # Generate Dissimilar Pairs (Label 1)
    while len(song_pairs) < 2 * num_pairs:
        folder1, folder2 = random.sample(all_folders, 2)  #ensures two different folders
        song1 = random.choice(data[folder1])
        song2 = random.choice(data[folder2])
        song_pairs.append({
            "Pair": [f"{folder1}/{song1}", f"{folder2}/{song2}"],
            "Label": 1
        })
    random.shuffle(song_pairs)

    return song_pairs

def remove_duplicate_pairs(pairs):
    seen = set()
    unique_pairs = []

    for pair in pairs:
        pair_set = frozenset(pair["Pair"])  # Order-independent check
        if pair_set not in seen:
            seen.add(pair_set)
            unique_pairs.append(pair)
    return unique_pairs

def check_pair_balance(pairs):
    label_counts = {0: 0, 1: 0}
    
    for pair in pairs:
        label_counts[pair["Label"]] += 1

    print(f"Similar Pairs (Label 0): {label_counts[0]}")
    print(f"Dissimilar Pairs (Label 1): {label_counts[1]}")


def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Generate song pairs from dataset")
    parser.add_argument('--parent_folder', type=str, required=True, help="Path to the parent folder containing subfolders with data.")
    parser.add_argument('--num_pairs', type=int, required=True, help="Number of pairs to generate.")

    args = parser.parse_args()
    data = get_all_folders(args.parent_folder)
    pairs_list = create_song_pairs(data, num_pairs=args.num_pairs)
    pairs_list_no_duplicates = remove_duplicate_pairs(pairs_list)
    with open("song_pairs.json", "w") as f:
        json.dump(pairs_list_no_duplicates, f, indent=4)
        
    check_pair_balance(pairs_list_no_duplicates)
    
    print(f"Generated {int(len(pairs_list_no_duplicates)/2)} song pairs and saved to 'song_pairs.json'.") # the length of the list is twice the number of pairs


if __name__ == "__main__":
    main()


#usage
#python mapping.py --parent_folder "../data/17K_data" --num_pairs 10000