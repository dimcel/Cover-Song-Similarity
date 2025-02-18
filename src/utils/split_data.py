# import os
# import h5py
# import json
# import argparse
# import numpy as np
# from sklearn.model_selection import train_test_split

# def load_hpcp_features(file_path):
#     """Loads HPCP features from an HDF5 file."""
#     with h5py.File(file_path, 'r') as f:
#         return np.array(f['hpcp'])

# def process_data(mapping_json, base_path, output_dir, batch_size, sample_ratio, test_size, val_size):
#     """Processes data from JSON mappings, splits into train/val/test, and stores in HDF5 format."""

#     with open(mapping_json, 'r') as f:
#         mappings = json.load(f)

#     if sample_ratio < 1.0:
#         np.random.shuffle(mappings)
#         mappings = mappings[:int(len(mappings) * sample_ratio)]

#     train_mappings, test_mappings = train_test_split(mappings, test_size=test_size, random_state=1)
#     train_mappings, val_mappings = train_test_split(train_mappings, test_size=val_size / (1 - test_size), random_state=1)

#     print(f"Dataset split: Train={len(train_mappings)}, Val={len(val_mappings)}, Test={len(test_mappings)}")

#     os.makedirs(output_dir, exist_ok=True)

#     def save_to_h5(mappings, feature_path, label_path):
#         first_pair = True
#         with h5py.File(feature_path, 'w') as f_feat, h5py.File(label_path, 'w') as f_label:
#             for i in range(0, len(mappings), batch_size):
#                 batch = mappings[i:i + batch_size]
                
#                 pairs, labels = [], []
                
#                 for item in batch:
#                     pair = item['Pair']
#                     label = item['Label']
                    
#                     file_1 = os.path.join(base_path, pair[0])
#                     file_2 = os.path.join(base_path, pair[1])
                    
#                     if os.path.exists(file_1) and os.path.exists(file_2):
#                         hpcp_1 = load_hpcp_features(file_1)
#                         hpcp_2 = load_hpcp_features(file_2)
#                         pairs.append(np.stack([hpcp_1, hpcp_2]))
#                         labels.append(label)
#                     else:
#                         print(f"Missing file(s): {file_1}, {file_2}")
                
#                 pairs = np.array(pairs)
#                 labels = np.array(labels)
                
#                 if first_pair:
#                     pair_dset = f_feat.create_dataset('pairs', data=pairs, maxshape=(None, *pairs.shape[1:]), dtype='float32')
#                     label_dset = f_label.create_dataset('labels', data=labels, maxshape=(None,), dtype='int')
#                     first_pair = False
#                 else:
#                     pair_dset.resize((pair_dset.shape[0] + pairs.shape[0], *pairs.shape[1:]))
#                     pair_dset[-pairs.shape[0]:] = pairs
                    
#                     label_dset.resize((label_dset.shape[0] + labels.shape[0],))
#                     label_dset[-labels.shape[0]:] = labels

#     # Save train, validation, and test sets
#     save_to_h5(train_mappings, os.path.join(output_dir, 'train_features.h5'), os.path.join(output_dir, 'train_labels.h5'))
#     save_to_h5(val_mappings, os.path.join(output_dir, 'val_features.h5'), os.path.join(output_dir, 'val_labels.h5'))
#     save_to_h5(test_mappings, os.path.join(output_dir, 'test_features.h5'), os.path.join(output_dir, 'test_labels.h5'))

#     print(f"Processed data saved in '{output_dir}'")

# def main():
#     """Parses command-line arguments and runs the script."""
#     parser = argparse.ArgumentParser(description="Process and split song pairs into train/val/test datasets.")
#     parser.add_argument('--mapping_json', type=str, required=True, help="Path to JSON file with song pairs.")
#     parser.add_argument('--base_path', type=str, required=True, help="Base directory containing HPCP data.")
#     parser.add_argument('--output_dir', type=str, required=True, help="Directory to save processed data.")
#     parser.add_argument('--batch_size', type=int, default=100, help="Batch size for processing (default: 100).")
#     parser.add_argument('--sample_ratio', type=float, default=1.0, help="Fraction of dataset to use (default: 1.0).")
#     parser.add_argument('--test_size', type=float, default=0.1, help="Proportion of test data (default: 0.1).")
#     parser.add_argument('--val_size', type=float, default=0.1, help="Proportion of validation data (default: 0.1).")

#     args = parser.parse_args()

#     process_data(
#         mapping_json=args.mapping_json,
#         base_path=args.base_path,
#         output_dir=args.output_dir,
#         batch_size=args.batch_size,
#         sample_ratio=args.sample_ratio,
#         test_size=args.test_size,
#         val_size=args.val_size
#     )

# if __name__ == "__main__":
#     main()

# # Example usage:
# # python splitting_data.py \
# #   --mapping_json "song_pairs.json" \
# #   --base_path "../data/17K_data" \
# #   --output_dir "../data/data_model" \
# #   --batch_size 100 \
# #   --sample_ratio 0.1 \
# #   --test_size 0.1 \
# #   --val_size 0.1

import os
import h5py
import json
import argparse
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

def load_hpcp_features(file_path):
    """Loads HPCP features from an HDF5 file."""
    with h5py.File(file_path, 'r') as f:
        return np.array(f['hpcp'])

def balanced_split(mappings, test_size, val_size):
    """Splits data into balanced train, validation, and test sets."""
    labels = [item['Label'] for item in mappings]
    
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=1)
    train_idx, test_idx = next(sss.split(mappings, labels))
    train_mappings = [mappings[i] for i in train_idx]
    test_mappings = [mappings[i] for i in test_idx]
    
    labels_train = [item['Label'] for item in train_mappings]
    val_size_adj = val_size / (1 - test_size)
    
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_size_adj, random_state=1)
    train_idx, val_idx = next(sss.split(train_mappings, labels_train))
    final_train_mappings = [train_mappings[i] for i in train_idx]
    val_mappings = [train_mappings[i] for i in val_idx]
    
    return final_train_mappings, val_mappings, test_mappings

def save_json(data, file_path):
    """Saves mappings and labels to a JSON file."""
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

def count_labels(label_path):
    """Counts occurrences of labels 0 and 1 in the label HDF5 file."""
    with h5py.File(label_path, 'r') as f:
        labels = f['labels'][:]
    unique, counts = np.unique(labels, return_counts=True)
    return dict(zip(unique, counts))

def process_data(mapping_json, base_path, output_dir, batch_size, sample_ratio, test_size, val_size):
    """Processes data from JSON mappings, splits into balanced train/val/test, and stores in HDF5 and JSON formats."""
    with open(mapping_json, 'r') as f:
        mappings = json.load(f)
    
    if sample_ratio < 1.0:
        np.random.shuffle(mappings)
        mappings = mappings[:int(len(mappings) * sample_ratio)]
    
    train_mappings, val_mappings, test_mappings = balanced_split(mappings, test_size, val_size)
    print(f"Dataset split: Train={len(train_mappings)}, Val={len(val_mappings)}, Test={len(test_mappings)}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    save_json(train_mappings, os.path.join(output_dir, 'train_data.json'))
    save_json(val_mappings, os.path.join(output_dir, 'val_data.json'))
    save_json(test_mappings, os.path.join(output_dir, 'test_data.json'))

    def save_to_h5(mappings, feature_path, label_path):
        first_pair = True
        with h5py.File(feature_path, 'w') as f_feat, h5py.File(label_path, 'w') as f_label:
            for i in range(0, len(mappings), batch_size):
                batch = mappings[i:i + batch_size]
                pairs, labels = [], []
                for item in batch:
                    pair, label = item['Pair'], item['Label']
                    file_1, file_2 = os.path.join(base_path, pair[0]), os.path.join(base_path, pair[1])
                    if os.path.exists(file_1) and os.path.exists(file_2):
                        hpcp_1, hpcp_2 = load_hpcp_features(file_1), load_hpcp_features(file_2)
                        pairs.append(np.stack([hpcp_1, hpcp_2]))
                        labels.append(label)
                    else:
                        print(f"Missing file(s): {file_1}, {file_2}")
                
                pairs, labels = np.array(pairs), np.array(labels)
                if first_pair:
                    pair_dset = f_feat.create_dataset('pairs', data=pairs, maxshape=(None, *pairs.shape[1:]), dtype='float32')
                    label_dset = f_label.create_dataset('labels', data=labels, maxshape=(None,), dtype='int')
                    first_pair = False
                else:
                    pair_dset.resize((pair_dset.shape[0] + pairs.shape[0], *pairs.shape[1:]))
                    pair_dset[-pairs.shape[0]:] = pairs
                    label_dset.resize((label_dset.shape[0] + labels.shape[0],))
                    label_dset[-labels.shape[0]:] = labels
    
    save_to_h5(train_mappings, os.path.join(output_dir, 'train_features.h5'), os.path.join(output_dir, 'train_labels.h5'))
    save_to_h5(val_mappings, os.path.join(output_dir, 'val_features.h5'), os.path.join(output_dir, 'val_labels.h5'))
    save_to_h5(test_mappings, os.path.join(output_dir, 'test_features.h5'), os.path.join(output_dir, 'test_labels.h5'))
    
    print("Label distributions:")
    print("Train:", count_labels(os.path.join(output_dir, 'train_labels.h5')))
    print("Validation:", count_labels(os.path.join(output_dir, 'val_labels.h5')))
    print("Test:", count_labels(os.path.join(output_dir, 'test_labels.h5')))
    
    print(f"Processed data saved in '{output_dir}'")


def main():
    """Parses command-line arguments and runs the script."""
    parser = argparse.ArgumentParser(description="Process and split song pairs into train/val/test datasets.")
    parser.add_argument('--mapping_json', type=str, required=True, help="Path to JSON file with song pairs.")
    parser.add_argument('--base_path', type=str, required=True, help="Base directory containing HPCP data.")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save processed data.")
    parser.add_argument('--batch_size', type=int, default=100, help="Batch size for processing (default: 100).")
    parser.add_argument('--sample_ratio', type=float, default=1.0, help="Fraction of dataset to use (default: 1.0).")
    parser.add_argument('--test_size', type=float, default=0.1, help="Proportion of test data (default: 0.1).")
    parser.add_argument('--val_size', type=float, default=0.1, help="Proportion of validation data (default: 0.1).")

    args = parser.parse_args()

    process_data(
        mapping_json=args.mapping_json,
        base_path=args.base_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        sample_ratio=args.sample_ratio,
        test_size=args.test_size,
        val_size=args.val_size
    )


if __name__ == "__main__":
    main()