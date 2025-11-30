import numpy as np
import torch
from torch.utils.data import Dataset
import os

class TestDataset(Dataset):
    """Dataset for loading test data from .npy files"""
    def __init__(self, device, data_path='./data/data_test'):
        self.device = device
        self.data_path = data_path
        
        # Load test data and convert to float32
        images_raw = np.load(f'{data_path}/images_ny.npy')
        alphas = np.load(f'{data_path}/alphas.npy')
        
        # Normalize images by dividing by alpha (to get [0,1] range)
        self.images_ny = torch.from_numpy(images_raw / alphas[:, None, None, None, None]).to(torch.float32).to(device)
        self.depth_maps = torch.from_numpy(np.load(f'{data_path}/depth_maps.npy')).to(torch.float32).to(device)
        
        print(f'Loaded test dataset from {data_path}')
        print(f'Images shape: {self.images_ny.shape}')
        print(f'Images range: [{self.images_ny.min():.4f}, {self.images_ny.max():.4f}]')
        print(f'Depth maps shape: {self.depth_maps.shape}')
    
    def __len__(self):
        return self.images_ny.shape[0]
    
    def __getitem__(self, idx):
        return self.images_ny[idx], self.depth_maps[idx]


class ShapeDataset(Dataset):
    """Dataset for loading training/validation data from .npy files"""
    def __init__(self, device, data_path='./data/data_train_val', mode='train'):
        self.device = device
        self.data_path = data_path
        self.mode = mode
        
        # Load data based on mode
        if mode == 'train':
            data_file = f'{data_path}/train_data.npy'
        elif mode == 'val':
            data_file = f'{data_path}/val_data.npy'
        else:
            raise ValueError(f"Mode must be 'train' or 'val', got {mode}")
        
        if os.path.exists(data_file):
            self.data = torch.from_numpy(np.load(data_file)).to(device)
            print(f'Loaded {mode} dataset from {data_file}')
            print(f'Data shape: {self.data.shape}')
        else:
            print(f'Warning: {data_file} not found')
            self.data = None
    
    def __len__(self):
        return self.data.shape[0] if self.data is not None else 0
    
    def __getitem__(self, idx):
        if self.data is not None:
            return self.data[idx]
        else:
            raise ValueError("Dataset not loaded")
