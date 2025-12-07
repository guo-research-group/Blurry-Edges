"""
Training Script for Depth Densifier
====================================
Trains lightweight U-Net to densify sparse Blurry-Edges depth maps.

Strategy:
1. Freeze entire Blurry-Edges pipeline
2. Use pre-computed sparse depths as input
3. Train only the densifier network
4. Use small crops (256x256) to fit in 4GB GPU memory
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from models.depth_densifier import DepthDensifierUNet, DepthDensifierLoss

# Import compute_errors directly to avoid cv2 dependency
import sys
sys.path.append('./utils')
try:
    from utils.metrics import compute_errors
except ImportError:
    # Fallback if cv2 not available
    def compute_errors(gt, pred):
        """Compute depth metrics without cv2 dependency"""
        thresh = np.maximum((gt / pred), (pred / gt))
        delta1 = (thresh < 1.25).mean()
        rmse = np.sqrt(((gt - pred) ** 2).mean())
        abs_rel = (np.abs(gt - pred) / gt).mean()
        return {'delta1': delta1, 'rmse': rmse, 'abs_rel': abs_rel}


class DensifierDataset(Dataset):
    """
    Dataset for training depth densifier
    Loads pre-computed sparse depth maps from Blurry-Edges
    
    Data split (200 images total):
    - Train: indices 0-139 (140 images, 70%)
    - Val: indices 140-179 (40 images, 20%)
    - Test: indices 180-199 (20 images, 10%) - HELD OUT, never used in training
    """
    def __init__(self, data_path, mode='train', crop_size=None, indices=None):
        """
        Args:
            data_path: Path to data directory
            mode: 'train', 'val', or 'test'
            crop_size: Random crop size (H, W) or None for full image
            indices: List of indices to use (for train/val/test split)
        """
        self.data_path = data_path
        self.mode = mode
        self.crop_size = crop_size
        
        # Load data
        self.images = np.load(os.path.join(data_path, 'images_ny.npy'))  # [N, 2, H, W, 3]
        # Note: alphas.npy contains scalar values, not depth maps
        # We'll use depth_maps.npy for ground truth if available, otherwise compute from data
        depth_maps_path = os.path.join(data_path, 'depth_maps.npy')
        if os.path.exists(depth_maps_path):
            self.gt_depths = np.load(depth_maps_path)  # [N, H, W]
        else:
            # Fall back to alphas (may need to be reshaped/converted)
            self.gt_depths = None
        
        # Apply index filtering if specified
        if indices is not None:
            self.images = self.images[indices]
            if self.gt_depths is not None:
                self.gt_depths = self.gt_depths[indices]
            self.index_offset = indices[0]  # For loading correct depth files
        else:
            self.index_offset = 0
        
        # Load pre-computed Blurry-Edges outputs
        sparse_dir = './logs/blurry_edges_depths'
        
        self.sparse_depths = []
        self.confidence_maps = []
        self.boundary_maps = []
        
        # Load all saved depth maps
        for i in range(len(self.images)):
            # Use correct file index (accounting for train/val/test split)
            file_idx = i + self.index_offset
            depth_path = f'{sparse_dir}/depth_{file_idx:03d}.npy'
            conf_path = f'{sparse_dir}/confidence_{file_idx:03d}.npy'
            boundary_path = f'{sparse_dir}/boundary_{file_idx:03d}.npy'
            
            if not os.path.exists(depth_path):
                raise FileNotFoundError(f"Depth file not found: {depth_path}\n"
                                       f"Please run: python blurry_edges_test.py first to generate depth maps for all 200 images")
            
            depth = np.load(depth_path)
            if len(depth.shape) == 3:
                depth = depth[0]
            self.sparse_depths.append(depth)
            
            # Load confidence
            if os.path.exists(conf_path):
                conf = np.load(conf_path)
                if len(conf.shape) == 3:
                    conf = conf[0]
                self.confidence_maps.append(conf)
            else:
                self.confidence_maps.append(np.ones_like(depth) * 0.5)
            
            # Load or compute boundary map
            if os.path.exists(boundary_path):
                boundary = np.load(boundary_path)
                if len(boundary.shape) == 3:
                    boundary = boundary[0]
                self.boundary_maps.append(boundary)
            else:
                # Compute simple boundary from depth
                boundary = self._compute_boundary(depth)
                self.boundary_maps.append(boundary)
        
        print(f"Loaded {len(self.sparse_depths)} samples for {mode}")
    
    def _compute_boundary(self, depth):
        """Compute boundary map from depth using Sobel filter"""
        from scipy.ndimage import sobel
        dx = sobel(depth, axis=0)
        dy = sobel(depth, axis=1)
        boundary = np.sqrt(dx**2 + dy**2)
        boundary = (boundary - boundary.min()) / (boundary.max() - boundary.min() + 1e-8)
        return boundary
    
    def __len__(self):
        return len(self.sparse_depths)
    
    def __getitem__(self, idx):
        """
        Returns:
            input_tensor: [6, H, W] (sparse_depth, boundary, confidence, RGB)
            gt_depth: [1, H, W] (ground truth dense depth)
            valid_mask: [1, H, W] (valid pixel mask)
        """
        try:
            # Get data
            image = self.images[idx, 0, :, :, :]  # Use first view [H, W, 3]
            sparse_depth = self.sparse_depths[idx]  # [H, W]
            confidence = self.confidence_maps[idx]  # [H, W]
            boundary = self.boundary_maps[idx]      # [H, W]
            
            # Use ground truth depth maps if available, otherwise use sparse depth as gt
            if self.gt_depths is not None:
                gt_depth = self.gt_depths[idx]  # [H, W]
            else:
                # Fallback: use the saved sparse depth as "ground truth" (not ideal but works)
                gt_depth = sparse_depth.copy()
        except IndexError as e:
            raise IndexError(f"Index {idx} out of bounds. Dataset has {len(self.images)} images, "
                           f"{len(self.sparse_depths)} sparse depths. Index offset: {self.index_offset}") from e
        
        # Normalize image to [0, 1]
        if image.max() > 1.0:
            image = image / 255.0
        
        # Create valid mask (where ground truth exists)
        valid_mask = (gt_depth > 0).astype(np.float32)
        
        # Random crop if specified
        if self.crop_size is not None:
            h, w = sparse_depth.shape
            crop_h, crop_w = self.crop_size
            
            if h > crop_h and w > crop_w:
                top = np.random.randint(0, h - crop_h)
                left = np.random.randint(0, w - crop_w)
                
                image = image[top:top+crop_h, left:left+crop_w, :]
                sparse_depth = sparse_depth[top:top+crop_h, left:left+crop_w]
                confidence = confidence[top:top+crop_h, left:left+crop_w]
                boundary = boundary[top:top+crop_h, left:left+crop_w]
                gt_depth = gt_depth[top:top+crop_h, left:left+crop_w]
                valid_mask = valid_mask[top:top+crop_h, left:left+crop_w]
        
        # Validate shapes before stacking
        if not isinstance(gt_depth, np.ndarray) or gt_depth.ndim != 2:
            raise ValueError(f"Invalid gt_depth shape: {gt_depth.shape if isinstance(gt_depth, np.ndarray) else type(gt_depth)}. "
                           f"Expected 2D array. Index: {idx}, mode: {self.mode}")
        
        # Stack inputs: [sparse, boundary, confidence, R, G, B]
        input_tensor = np.stack([
            sparse_depth,
            boundary,
            confidence,
            image[:, :, 0],
            image[:, :, 1],
            image[:, :, 2]
        ], axis=0).astype(np.float32)
        
        gt_depth = gt_depth[np.newaxis, :, :].astype(np.float32)
        valid_mask = valid_mask[np.newaxis, :, :].astype(np.float32)
        
        return input_tensor, gt_depth, valid_mask


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    
    total_loss = 0.0
    total_l1 = 0.0
    total_smooth = 0.0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for inputs, gt_depth, valid_mask in pbar:
        inputs = inputs.to(device)
        gt_depth = gt_depth.to(device)
        valid_mask = valid_mask.to(device)
        
        # Get RGB for smoothness loss
        image = inputs[:, 3:6, :, :]
        
        # Forward pass
        pred_depth = model(inputs)
        
        # Compute loss
        loss, l1_loss, smooth_loss = criterion(pred_depth, gt_depth, image, valid_mask)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Accumulate losses
        total_loss += loss.item()
        total_l1 += l1_loss.item()
        total_smooth += smooth_loss.item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'l1': f'{l1_loss.item():.4f}',
            'smooth': f'{smooth_loss.item():.4f}'
        })
    
    n = len(dataloader)
    return total_loss / n, total_l1 / n, total_smooth / n


def validate(model, dataloader, criterion, device):
    """Validate model"""
    model.eval()
    
    total_loss = 0.0
    total_l1 = 0.0
    total_smooth = 0.0
    
    metrics = {'delta1': [], 'rmse': [], 'abs_rel': []}
    
    with torch.no_grad():
        for inputs, gt_depth, valid_mask in tqdm(dataloader, desc="Validation"):
            inputs = inputs.to(device)
            gt_depth = gt_depth.to(device)
            valid_mask = valid_mask.to(device)
            
            # Get RGB for smoothness loss
            image = inputs[:, 3:6, :, :]
            
            # Forward pass
            pred_depth = model(inputs)
            
            # Compute loss
            loss, l1_loss, smooth_loss = criterion(pred_depth, gt_depth, image, valid_mask)
            
            total_loss += loss.item()
            total_l1 += l1_loss.item()
            total_smooth += smooth_loss.item()
            
            # Compute metrics
            pred_np = pred_depth.cpu().numpy()
            gt_np = gt_depth.cpu().numpy()
            valid_np = valid_mask.cpu().numpy()
            
            for b in range(pred_np.shape[0]):
                mask = valid_np[b, 0] > 0
                if mask.sum() > 0:
                    result = compute_errors(gt_np[b, 0][mask], pred_np[b, 0][mask])
                    metrics['delta1'].append(result['delta1'])
                    metrics['rmse'].append(result['rmse'])
                    metrics['abs_rel'].append(result['abs_rel'])
    
    n = len(dataloader)
    avg_metrics = {k: np.mean(v) for k, v in metrics.items()}
    
    return total_loss / n, total_l1 / n, total_smooth / n, avg_metrics


def main():
    parser = argparse.ArgumentParser(description='Train Depth Densifier')
    parser.add_argument('--data_path', type=str, default='./data_test/regular',
                        help='Path to training data')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size (keep small for 4GB GPU)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--crop_size', type=int, default=None,
                        help='Random crop size (None = full image)')
    parser.add_argument('--lambda_l1', type=float, default=1.0,
                        help='Weight for L1 loss')
    parser.add_argument('--lambda_smooth', type=float, default=0.1,
                        help='Weight for smoothness loss')
    parser.add_argument('--save_dir', type=str, default='./pretrained_weights',
                        help='Directory to save checkpoints')
    parser.add_argument('--cuda', type=str, default='cuda:0',
                        help='CUDA device')
    
    args = parser.parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Device
    device = torch.device(args.cuda if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets with proper train/val/test split
    # Total 200 images: Train 0-139, Val 140-179, Test 180-199 (HELD OUT)
    crop_size = (args.crop_size, args.crop_size) if args.crop_size else None
    
    print("\n" + "="*60)
    print("DATA SPLIT (No Data Leakage)")
    print("="*60)
    print("Train: indices 0-139 (140 images, 70%)")
    print("Val:   indices 140-179 (40 images, 20%)")
    print("Test:  indices 180-199 (20 images, 10%) - HELD OUT")
    print("="*60 + "\n")
    
    # Create index lists
    train_indices = list(range(0, 140))      # 0-139: 140 images
    val_indices = list(range(140, 180))      # 140-179: 40 images
    # test_indices = list(range(180, 200))   # 180-199: 20 images (NOT USED HERE)
    
    train_dataset = DensifierDataset(args.data_path, mode='train', crop_size=crop_size, indices=train_indices)
    val_dataset = DensifierDataset(args.data_path, mode='val', crop_size=None, indices=val_indices)  # No crop for val
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                               shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=2, pin_memory=True)
    
    # Create model
    model = DepthDensifierUNet(in_channels=6, out_channels=1)
    model = model.to(device)
    
    print(f"Model parameters: {model.count_parameters():,}")
    
    # Loss and optimizer
    criterion = DepthDensifierLoss(lambda_l1=args.lambda_l1, lambda_smooth=args.lambda_smooth)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                       factor=0.5, patience=10, verbose=True)
    
    # Training loop
    best_rmse = float('inf')
    
    print("\n" + "="*60)
    print("Starting Training")
    print("="*60)
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 60)
        
        # Train
        train_loss, train_l1, train_smooth = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        print(f"Train - Loss: {train_loss:.4f}, L1: {train_l1:.4f}, Smooth: {train_smooth:.4f}")
        
        # Validate
        val_loss, val_l1, val_smooth, val_metrics = validate(
            model, val_loader, criterion, device
        )
        
        print(f"Val   - Loss: {val_loss:.4f}, L1: {val_l1:.4f}, Smooth: {val_smooth:.4f}")
        print(f"Val   - RMSE: {val_metrics['rmse']:.4f}, delta1: {val_metrics['delta1']:.4f}, "
              f"AbsRel: {val_metrics['abs_rel']:.4f}")
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save best model
        if val_metrics['rmse'] < best_rmse:
            best_rmse = val_metrics['rmse']
            save_path = os.path.join(args.save_dir, 'best_densifier.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'rmse': best_rmse,
                'metrics': val_metrics
            }, save_path)
            print(f"✓ Saved best model (RMSE: {best_rmse:.4f})")
        
        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            save_path = os.path.join(args.save_dir, f'densifier_epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, save_path)
    
    print("\n" + "="*60)
    print(f"Training completed! Best RMSE: {best_rmse:.4f}")
    print("="*60)


if __name__ == '__main__':
    main()
