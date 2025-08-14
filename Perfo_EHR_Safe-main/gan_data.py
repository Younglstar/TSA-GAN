# gan_data.py
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple
import pickle

class EncodedDataset(Dataset):
    def __init__(self, data: np.ndarray):
        """Dataset for encoded EHR data"""
        self.data = torch.FloatTensor(data)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def load_encoded_data(encoded_data_path: str) -> np.ndarray:
    """Load encoded data from file"""
    with open(encoded_data_path, 'rb') as f:
        data = pickle.load(f)
    return data

def create_dataloader(
    data: np.ndarray,
    batch_size: int,
    shuffle: bool = True
) -> torch.utils.data.DataLoader:
    """Create dataloader for encoded data"""
    dataset = EncodedDataset(data)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True  # Important for batch normalization
    )

def scale_data(data: np.ndarray) -> Tuple[np.ndarray, dict]:
    """Scale data to [-1, 1] range for GAN training"""
    # Calculate scaling parameters
    data_min = np.min(data, axis=0)
    data_max = np.max(data, axis=0)
    
    # Handle constant features
    scale = data_max - data_min
    scale[scale == 0] = 1  # Avoid division by zero
    
    # Scale to [-1, 1]
    scaled_data = 2 * (data - data_min) / scale - 1
    
    scaling_params = {
        'min': data_min,
        'max': data_max,
        'scale': scale
    }
    
    return scaled_data, scaling_params

def inverse_scale_data(
    scaled_data: np.ndarray,
    scaling_params: dict
) -> np.ndarray:
    """Inverse scale data back to original range"""
    return (scaled_data + 1) / 2 * scaling_params['scale'] + scaling_params['min']