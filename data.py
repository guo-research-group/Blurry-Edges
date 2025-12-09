import os
import numpy as np
import torch
from torch.utils.data import Dataset

class ShapeDataset(Dataset):
    def __init__(self, device, data_path='.', train=False, mode='local'):
        partition = 'train' if train else 'val'
        self.mode = mode
        if self.mode == 'local':
            img_ny = np.load(f'{data_path}/patches_ny_{partition}.npy')
            img_gt = np.load(f'{data_path}/patches_gt_{partition}.npy')
            alpha = np.load(f'{data_path}/alphas_{partition}.npy')
            bndry_dist = np.load(f'{data_path}/boundary_distances_{partition}.npy')
            deri = np.load(f'{data_path}/derivative_maps_{partition}.npy')
            self.img_gt = torch.from_numpy(img_gt).float()
            self.bndry_dist = torch.from_numpy(bndry_dist).float()
            self.deri = torch.from_numpy(deri[:, 1:-1, 1:-1, :]).float()
        elif self.mode == 'global_pre':
            img_ny = np.load(f'{data_path}/images_ny_{partition}.npy')
            alpha = np.load(f'{data_path}/alphas_{partition}.npy')
        elif self.mode == 'global':
            input_param = np.load(f'{data_path}/params_src_{partition}.npy')
            img_ny = np.load(f'{data_path}/images_ny_{partition}.npy')
            img_gt = np.load(f'{data_path}/images_gt_{partition}.npy')
            deri = np.load(f'{data_path}/derivative_maps_{partition}.npy')
            bndry_dist = np.load(f'{data_path}/boundary_distances_{partition}.npy')
            bndry_depth = np.load(f'{data_path}/boundary_depths_{partition}.npy')
            alpha = np.load(f'{data_path}/alphas_{partition}.npy')
            self.input_param = torch.from_numpy(input_param).float()
            self.img_gt = torch.from_numpy(img_gt).float()
            self.deri = torch.from_numpy(deri[:, :, 1:-1, 1:-1, :]).float()
            self.bndry_dist = torch.from_numpy(bndry_dist).float()
            self.bndry_depth = torch.from_numpy(bndry_depth).float()
        self.img_ny = torch.from_numpy(img_ny).float()
        self.alpha = torch.from_numpy(alpha).float()
        self.device = device
    def __len__(self):
        return self.img_ny.shape[0]
    def __getitem__(self, idx):
        img_ny = self.img_ny[idx, ...].to(self.device)
        alpha = self.alpha[idx].to(self.device)
        if self.mode == 'local':
            img_gt = self.img_gt[idx, ...].to(self.device)
            bndry_dist = self.bndry_dist[idx, ...].to(self.device)
            deri = self.deri[idx, ...].to(self.device)
            return img_ny/alpha, img_gt/alpha, bndry_dist, deri
        elif self.mode == 'global_pre':
            return img_ny/alpha
        elif self.mode == 'global':
            input_param = self.input_param[idx, ...].to(self.device)
            img_gt = self.img_gt[idx, ...].to(self.device)
            deri = self.deri[idx, ...].to(self.device)
            bndry_dist = self.bndry_dist[idx, ...].to(self.device)
            bndry_depth = self.bndry_depth[idx, ...].to(self.device)
            return input_param, img_ny/alpha, img_gt/alpha, bndry_dist, deri, bndry_depth
    
class TestDataset(Dataset):
    def __init__(self, device, data_path='.'):
        ny_img = np.load(os.path.join(data_path, 'images_ny.npy'))
        depth_map = np.load(os.path.join(data_path, 'depth_maps.npy'))
        alpha = np.load(os.path.join(data_path, 'alphas.npy'))
        self.ny_img = torch.from_numpy(ny_img).float()
        self.depth_map = torch.from_numpy(depth_map).float()
        self.alpha = torch.from_numpy(alpha).float()
        self.device = device
    def __len__(self):
        return self.ny_img.shape[0]
    def __getitem__(self, idx):
        ny_img = self.ny_img[idx,...].to(self.device)
        depth_map = self.depth_map[idx,:,:].to(self.device)
        alpha = self.alpha[idx].to(self.device)
        return ny_img/alpha, depth_map