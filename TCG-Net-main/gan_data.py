# gan_data.py
import torch
from scipy.interpolate import CubicSpline
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple
import pickle

from tqdm import tqdm


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
        data,
        batch_size: int,
        shuffle: bool = True
) -> torch.utils.data.DataLoader:
    """
    Create dataloader for GAN training
    data 可以是 numpy array 或 torch Dataset
    """
    if isinstance(data, np.ndarray):
        dataset = EncodedDataset(data)
    else:
        dataset = data  # 已经是 Dataset 的情况（如 TimeSeriesDataset）

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True
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


class TimeSeriesDataset(Dataset):
    def __init__(self, data, augment_factor=1, noise_std=0.01):
        """
        :param data: numpy array, shape (N, seq_len, features) 或 (N, seq_len)
        :param augment_factor: 数据复制倍数
        :param noise_std: 增强时添加的高斯噪声标准差
        """
        self.data = data
        self.augment_factor = augment_factor
        self.noise_std = noise_std
        self.samples = self._augment_data()

    def _augment_data(self):
        """
        仅做样本复制 + 轻度高斯噪声，不做扭曲
        """
        augmented = []
        for _ in range(self.augment_factor):
            noisy = self.data + np.random.normal(0, self.noise_std, self.data.shape)
            augmented.append(noisy)
        return np.concatenate(augmented, axis=0)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return torch.tensor(self.samples[idx], dtype=torch.float32)