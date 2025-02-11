# import sys
# sys.path.append("../tmp")
# import pytest
# import json
# import h5py
# import numpy as np
# from pathlib import Path
# from utils.split_data import process_data, load_hpcp_features


import pytest
import json
import h5py
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
#sys.path.append("../Cover-Song-Similarity")
from src.utils.split_data import process_data, load_hpcp_features

def test_load_hpcp_features(tmp_path):
    """Test loading HPCP features from an HDF5 file."""
    
    h5_file = tmp_path / "test.h5"
    
    hpcp_data = np.random.rand(100, 12)
    
    with h5py.File(h5_file, "w") as f:
        f.create_dataset("hpcp", data=hpcp_data)
    loaded_data = load_hpcp_features(h5_file)

    assert isinstance(loaded_data, np.ndarray), "Loaded data should be a NumPy array."
    assert loaded_data.shape == hpcp_data.shape, "Loaded data shape should match the original."
    assert np.allclose(loaded_data, hpcp_data), "Loaded data values should match the original."

def test_load_hpcp_features_missing_file():
    """Test handling of a missing file."""
    with pytest.raises(FileNotFoundError):
        load_hpcp_features(Path("non_existent.h5"))

def test_load_hpcp_features_missing_hpcp_key(tmp_path):
    """Test handling of a missing 'hpcp' dataset."""
    
    h5_file = tmp_path / "test_missing_key.h5"

    with h5py.File(h5_file, "w") as f:
        f.create_dataset("wrong_key", data=np.random.rand(50, 12))  #wrong key

    with pytest.raises(KeyError):
        load_hpcp_features(h5_file)


def test_process_data(tmp_path):
    """Test process_data with a valid dataset split."""
    
    mapping_json = tmp_path / "mappings.json"
    mappings = [
        {"Pair": ["folder1/file1.h5", "folder1/file2.h5"], "Label": 0},
        {"Pair": ["folder2/file3.h5", "folder2/file4.h5"], "Label": 1},
        {"Pair": ["folder3/file5.h5", "folder3/file6.h5"], "Label": 0},
        {"Pair": ["folder4/file7.h5", "folder4/file8.h5"], "Label": 1},
        {"Pair": ["folder5/file9.h5", "folder5/file10.h5"], "Label": 0},
    ]

    with open(mapping_json, "w") as f:
        json.dump(mappings, f)

    output_dir = tmp_path / "output"

    process_data(
        mapping_json=str(mapping_json),
        base_path=str(tmp_path),
        output_dir=str(output_dir),
        batch_size=2,
        sample_ratio=1.0,
        test_size=0.2,
        val_size=0.2
    )

    assert (output_dir / "train_features.h5").exists()
    assert (output_dir / "train_labels.h5").exists()
    assert (output_dir / "val_features.h5").exists()
    assert (output_dir / "val_labels.h5").exists()
    assert (output_dir / "test_features.h5").exists()
    assert (output_dir / "test_labels.h5").exists()

def test_process_data_missing_files(tmp_path):
    """Test process_data with missing files (should print warnings but not fail)."""
    
    mapping_json = tmp_path / "mappings.json"
    mappings = [
        {"Pair": ["folder1/file1.h5", "folder1/file2.h5"], "Label": 0},
        {"Pair": ["folder2/file3.h5", "folder2/file4.h5"], "Label": 1},
        {"Pair": ["folder3/file5.h5", "folder3/file6.h5"], "Label": 0},
        {"Pair": ["folder4/file7.h5", "folder4/file8.h5"], "Label": 1},
        {"Pair": ["folder5/file9.h5", "missing_folder/missing_file.h5"], "Label": 0},  #missing file
    ]

    with open(mapping_json, "w") as f:
        json.dump(mappings, f)

    output_dir = tmp_path / "output"

    process_data(
        mapping_json=str(mapping_json),
        base_path=str(tmp_path),
        output_dir=str(output_dir),
        batch_size=2,
        sample_ratio=1.0,
        test_size=0.2,
        val_size=0.2
    )
    assert (output_dir / "train_features.h5").exists()

