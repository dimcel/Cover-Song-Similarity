import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
#sys.path.append("../Cover-Song-Similarity")
import numpy as np
import pytest
from unittest.mock import patch
from src.preprocessing import preprocess_input, extract_hpcp, validate_array


def test_preprocess_input_wav():
    """Test preprocess_input with a WAV file using mocking."""
    with patch("src.preprocessing.extract_hpcp") as mock_extract_hpcp:
        mock_extract_hpcp.return_value = np.zeros((17000, 12))
        
        result = preprocess_input("dummy.wav")
        
        assert result.shape == (17000, 12)
        mock_extract_hpcp.assert_called_once_with("dummy.wav")


def test_preprocess_input_array():
    """Test preprocess_input with a valid NumPy array."""
    valid_array = np.random.rand(17000, 12)
    result = preprocess_input(valid_array)
    assert result.shape == (17000, 12)


def test_preprocess_input_list():
    """Test preprocess_input with a list (should convert to NumPy array)."""
    valid_list = [[0] * 12] * 17000
    result = preprocess_input(valid_list)
    assert isinstance(result, np.ndarray)
    assert result.shape == (17000, 12)


def test_preprocess_input_invalid():
    """Test preprocess_input with invalid inputs (should raise ValueError)."""
    with pytest.raises(ValueError):
        preprocess_input(123)  #invalid type

    with pytest.raises(ValueError):
        preprocess_input({"invalid": "data"})  #invalid dict input


def test_extract_hpcp():
    """Mock librosa functions to speed up extract_hpcp test."""
    with patch("librosa.load") as mock_load, patch("librosa.feature.chroma_cqt") as mock_hpcp:

        mock_load.return_value = (np.zeros(22050), 22050)  #1 second of silence
        
        # Mock chroma_cqt to return a small matrix
        mock_hpcp.return_value = np.random.rand(12, 16000)  #fake feature matrix
        
        result = extract_hpcp("dummy.wav")
        
        assert result.shape == (17000, 12)
        mock_load.assert_called_once_with("dummy.wav", sr=22050)
        mock_hpcp.assert_called_once()


def test_validate_array():
    """Test validate_array function for padding and truncation."""
    valid_array = np.random.rand(17000, 12) #correct shape
    assert validate_array(valid_array).shape == (17000, 12)

    #too short (should be padded)
    short_array = np.random.rand(16000, 12)
    assert validate_array(short_array).shape == (17000, 12)

    #too long (should be truncated)
    long_array = np.random.rand(18000, 12)
    assert validate_array(long_array).shape == (17000, 12)

