import sys
import pytest
import h5py
import numpy as np
import tempfile
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
#sys.path.append("../Cover-Song-Similarity")
from src.utils.resize_features import process_files

@pytest.fixture
def create_mock_h5():
    """Creates a temporary HDF5 file for testing."""
    def _create_h5(file_path: Path, num_frames: int, num_pitch_classes: int = 12):
        with h5py.File(file_path, "w") as h5f:
            hpcp_data = np.random.rand(num_frames, num_pitch_classes)
            h5f.create_dataset("hpcp", data=hpcp_data)
    return _create_h5

def test_process_files(create_mock_h5):
    target_length = 100 

    with tempfile.TemporaryDirectory() as source_dir, tempfile.TemporaryDirectory() as target_dir:
        source_path = Path(source_dir)
        target_path = Path(target_dir)

        work_folder = source_path / "work1"
        work_folder.mkdir()

        file1 = work_folder / "file1.h5"
        file2 = work_folder / "file2.h5"
        file3 = work_folder / "file3.h5"

        create_mock_h5(file1, 50)   #too short (should be padded)
        create_mock_h5(file2, 100)  #exactly correct
        create_mock_h5(file3, 150)  #too long (should be truncated)

        process_files(str(source_path), str(target_path), target_length)

        new_work_path = target_path / "work1"
        assert new_work_path.exists(), "Processed folder was not created"

        for file in [file1.name, file2.name, file3.name]:
            new_file_path = new_work_path / file
            assert new_file_path.exists(), f"Processed file {file} was not created"

            #check HPCP length
            with h5py.File(new_file_path, "r") as h5f:
                assert "hpcp" in h5f, f"{file} is missing 'hpcp' dataset"
                assert h5f["hpcp"].shape[0] == target_length, f"{file} has incorrect length"

def test_process_empty_folder():
    """Test when source folder is empty."""
    with tempfile.TemporaryDirectory() as source_dir, tempfile.TemporaryDirectory() as target_dir:
        process_files(source_dir, target_dir, 100)
        assert not list(Path(target_dir).iterdir()), "Target directory should be empty when source has no files"

def test_process_invalid_h5():
    """Test handling of corrupted .h5 files."""
    with tempfile.TemporaryDirectory() as source_dir, tempfile.TemporaryDirectory() as target_dir:
        work_folder = Path(source_dir) / "work1"
        work_folder.mkdir()

        bad_file = work_folder / "corrupt.h5"
        bad_file.write_text("This is not a valid HDF5 file")  #create an invalid file

        process_files(source_dir, target_dir, 100)

        assert not (Path(target_dir) / "work1" / "corrupt.h5").exists(), "Corrupt file should not be copied"