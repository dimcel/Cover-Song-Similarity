import h5py
import torch
from torch.utils.data import Dataset

class HpcpDataset(Dataset):
    def __init__(self, features_file, labels_file):
        """Dataset for loading HPCP feature pairs from HDF5 files."""
        self.features_file = features_file
        self.labels_file = labels_file
        self.features_h5 = h5py.File(self.features_file, 'r')
        self.labels_h5 = h5py.File(self.labels_file, 'r')
        self.pairs = self.features_h5['pairs']
        self.labels = self.labels_h5['labels']

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        pair = torch.tensor(self.pairs[idx], dtype=torch.float32)  # Shape: (2, 17000, 12)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)  # Shape: (1,)
        return pair, label

    def close(self):
        """Ensure files are closed properly when no longer needed."""
        self.features_h5.close()
        self.labels_h5.close()
