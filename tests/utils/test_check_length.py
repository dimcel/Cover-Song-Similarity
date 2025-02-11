import sys
from pathlib import Path
import pytest
import h5py
import tempfile
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
#sys.path.append("../Cover-Song-Similarity")
from src.utils.check_length import check_hpcp_lengths

def create_mock_h5_file(file_path: Path, num_frames: int, num_pitch_classes: int = 12):
    """Creates a mock .h5 file with HPCP data."""
    with h5py.File(file_path, "w") as f:
        f.create_dataset("hpcp", data=np.random.rand(num_frames, num_pitch_classes))

@pytest.mark.parametrize("frame_lengths,expected_output", [
    ([100, 100, 120], ["Files have different HPCP frame lengths!", "2 files have HPCP frame length 100", "1 files have HPCP frame length 120"]),
    ([100, 100, 100], ["All files have the same HPCP frame length: 100"]),
])
def test_check_hpcp_lengths(frame_lengths, expected_output, capfd):
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        #create subdirectories and files
        for i, length in enumerate(frame_lengths):
            folder = temp_path / f"wid_{i}"
            folder.mkdir()
            create_mock_h5_file(folder / f"file{i}.h5", length)

        check_hpcp_lengths(str(temp_path))

        captured = capfd.readouterr().out
        for expected in expected_output:
            assert expected in captured