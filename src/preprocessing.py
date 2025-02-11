"""preprocessing.py - Contains functions to preprocess input data if it is a WAV file or a list/array of HPCP features for the api"""

import numpy as np
import librosa

def preprocess_input(input_data):
    """
    Ensures the input is a valid (17000, 12) array.
    If input is a WAV file, it extracts HPCP features.
    """
    if isinstance(input_data, str) and input_data.endswith(".wav"):
        return extract_hpcp(input_data)
    elif isinstance(input_data, list) or isinstance(input_data, np.ndarray):
        return validate_array(np.array(input_data))
    else:
        raise ValueError("Invalid input format. Provide a .wav file or a valid array.")

def extract_hpcp(wav_path):
    """ Extract HPCP features from a WAV file. """
    y, sr = librosa.load(wav_path, sr=22050)
    hpcp = librosa.feature.chroma_cqt(y=y, sr=sr)  
    return validate_array(hpcp.T)

def validate_array(arr):
    """ Ensures the array is (17000, 12), padding/truncating if needed. """
    target_shape = (17000, 12)
    if arr.shape == target_shape:
        return arr
    elif arr.shape[0] < 17000:
        pad_width = ((0, 17000 - arr.shape[0]), (0, 0))
        return np.pad(arr, pad_width, mode='constant')
    else:
        return arr[:17000, :]  # Truncate
