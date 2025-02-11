import sys
from pathlib import Path
import pytest
import tempfile
import random

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
#sys.path.append("../Cover-Song-Similarity")
from unittest.mock import patch
from src.utils.make_mappings import get_all_folders, create_song_pairs

def test_get_all_folders():
    """Test that get_all_folders correctly lists files inside folders."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create folders with files
        folder1 = temp_path / "folder1"
        folder1.mkdir()
        (folder1 / "file1.txt").touch()
        (folder1 / "file2.txt").touch()

        folder2 = temp_path / "folder2"
        folder2.mkdir()
        (folder2 / "file3.txt").touch()

        # Create an empty folder
        empty_folder = temp_path / "empty_folder"
        empty_folder.mkdir()

        result = get_all_folders(str(temp_path))
        expected = {
            "folder1": ["file2.txt", "file1.txt"],
            "folder2": ["file3.txt"]
        }

        assert result == expected, f"Expected {expected}, but got {result}"

@pytest.fixture
def mock_data():
    """Creates a mock dataset with enough folders and songs."""
    return {
        "work1": ["song1.mp3", "song2.mp3", "song3.mp3"],
        "work2": ["song4.mp3", "song5.mp3"],
        "work3": ["song6.mp3", "song7.mp3"],
        "work4": ["song8.mp3", "song9.mp3"]
    }

def test_create_song_pairs(mock_data):
    """Ensures function works consistently with a fixed random seed."""
    num_pairs = min(2, len(mock_data) - 1)

    random.seed(42)
    pairs1 = create_song_pairs(mock_data, num_pairs)

    random.seed(42)
    pairs2 = create_song_pairs(mock_data, num_pairs)

    assert pairs1 == pairs2, "Function should return consistent results with fixed seed"