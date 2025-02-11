import requests
import numpy as np
import json
import h5py

# Paths to HCPC feature files
file1_path = "data/17K_data/W_18_hpcp/P_3849_hpcp.h5"
# file1_path = "data/17K_data/W_18_hpcp/P_52324_hpcp.h5"
# file2_path = "data/17K_data/W_26_hpcp/P_26_hpcp.h5"
# file2_path = "data/17K_data/W_18_hpcp/P_84554_hpcp.h5"
# file2_path = "data/17K_data/W_94_hpcp/P_94_hpcp.h5"
file2_path = "data/17K_data/W_48_hpcp/P_49_hpcp.h5"
# file2_path = "data/17K_data/W_148_hpcp/P_148_hpcp.h5"
# file2_path = "data/17K_data/W_202123_hpcp/P_792220_hpcp.h5"
# file2_path = "data/17K_data/W_31360_hpcp/P_31360_hpcp.h5"

def load_hpcp_from_h5(file_path):
    """Loads HCPC features from an HDF5 file."""
    with h5py.File(file_path, 'r') as f:
        return f['hpcp'][:]

# Load the HCPC features
input1 = load_hpcp_from_h5(file1_path).tolist() 
input2 = load_hpcp_from_h5(file2_path).tolist()

# API URL
url = "http://127.0.0.1:8000/predict/"

# Create request payload
data = {"input1": input1, "input2": input2}

# Send request
response = requests.post(url, json=data)

# Print response
print("Response:", response.json())

