"""
Depth Fusion: Combining Blurry-Edges (accurate boundaries) + MiDaS (dense coverage)

Energy minimization approach:
E(Z) = λ₁ * Σ F(x) * (Z(x) - Z_BE(x))²     [boundary term]
     + λ₂ * Σ (Z(x) - Z_mono(x))²          [data term]
     + λ₃ * Σ w_xy * (Z(x) - Z(y))²        [smoothness term]
"""
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter

class DepthFusion:
    def __init__(self, lambda1=10.0, lambda2=1.0, lambda3=0.1, 
                 num_iterations=50, device='cuda:0'):
        """
        Initialize depth fusion parameters
        
        Args:
            lambda1: Weight for boundary term (Blurry-Edges prior)
            lambda2: Weight for data term (monocular depth)
            lambda3: Weight for smoothness term
            num_iterations: Number of optimization iterations
            device: torch device
        """
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.num_iterations = num_iterations
        self.device = torch.device(device)
        
    def compute_edge_weights(self, image, sigma=1.0):
        """
        Compute edge-aware weights from image gradients
        
        Args:
            image: RGB image (H, W, 3) in range [0, 1]
            sigma: Gaussian smoothing parameter
            
        Returns:
            weights: Edge weights (H, W) - lower at edges, higher in smooth regions
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
        else:
            gray = image
            
        # Compute gradients
        grad_x = np.abs(np.gradient(gray, axis=1))
        grad_y = np.abs(np.gradient(gray, axis=0))
        gradient_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        # Edge weights: high gradient = low weight (preserve edges)
        weights = np.exp(-gradient_mag / sigma)
        return weights
    
    def align_midas_to_blurry_edges(self, Z_midas, Z_BE, confidence, threshold=0.1):
        """
        Align MiDaS depth to Blurry-Edges scale using confident regions
        
        MiDaS outputs inverse depth (disparity), we need to:
        1. Convert to metric depth
        2. Align scale to match Blurry-Edges
        
        Args:
            Z_midas: MiDaS disparity map (H, W)
            Z_BE: Blurry-Edges depth map (H, W)
            confidence: Confidence map from Blurry-Edges (H, W)
            threshold: Confidence threshold for alignment
            
        Returns:
            Z_midas_aligned: Aligned depth map
        """
        # Find confident regions in Blurry-Edges
        mask = confidence > threshold
        
        if mask.sum() < 10:
            # Not enough confident points, use global statistics
            print("Warning: Low confidence regions, using global alignment")
            mask = confidence > 0.01
        
        # MiDaS outputs inverse depth, convert to depth
        # Add small epsilon to avoid division by zero
        Z_midas_depth = 1.0 / (Z_midas + 1e-6)
        
        # Normalize to same range as Blurry-Edges
        # Find scale and shift using confident regions
        if mask.sum() > 0:
            midas_vals = Z_midas_depth[mask]
            be_vals = Z_BE[mask]
            
            # Robust scale estimation (median of ratios)
            scale = np.median(be_vals / (midas_vals + 1e-6))
            
            # Apply scale
            Z_midas_aligned = Z_midas_depth * scale
            
            # Optional: also estimate shift
            shift = np.median(be_vals - Z_midas_aligned[mask])
            Z_midas_aligned = Z_midas_aligned + shift
        else:
            # Fallback: simple normalization
            Z_midas_aligned = Z_midas_depth
            Z_midas_aligned = (Z_midas_aligned - Z_midas_aligned.min()) / (Z_midas_aligned.max() - Z_midas_aligned.min())
            Z_midas_aligned = Z_midas_aligned * (Z_BE.max() - Z_BE.min()) + Z_BE.min()
        
        return Z_midas_aligned
    
    def fuse_depths(self, Z_BE, Z_midas, confidence, image, verbose=False):
        """
        Fuse Blurry-Edges depth with MiDaS monocular depth
        
        Args:
            Z_BE: Blurry-Edges depth map (H, W)
            Z_midas: MiDaS depth map (H, W) - will be aligned
            confidence: Confidence map from Blurry-Edges (H, W)
            image: RGB image (H, W, 3) for edge-aware smoothing
            verbose: Print optimization progress
            
        Returns:
            Z_fused: Fused depth map (H, W)
        """
        if verbose:
            print(f"Input shapes: Z_BE={Z_BE.shape}, Z_midas={Z_midas.shape}, confidence={confidence.shape}")
        
        # Align MiDaS to Blurry-Edges scale
        Z_midas_aligned = self.align_midas_to_blurry_edges(Z_midas, Z_BE, confidence)
        
        if verbose:
            print(f"Alignment: MiDaS [{Z_midas.min():.2f}, {Z_midas.max():.2f}] "
                  f"-> [{Z_midas_aligned.min():.2f}, {Z_midas_aligned.max():.2f}]")
            print(f"Blurry-Edges: [{Z_BE.min():.2f}, {Z_BE.max():.2f}]")
        
        # Compute edge weights
        edge_weights = self.compute_edge_weights(image, sigma=0.1)
        
        # Initialize with weighted average
        total_weight = self.lambda1 * confidence + self.lambda2
        Z_fused = (self.lambda1 * confidence * Z_BE + self.lambda2 * Z_midas_aligned) / (total_weight + 1e-6)
        
        # Convert to torch for efficient computation
        Z_fused_torch = torch.from_numpy(Z_fused).float().to(self.device)
        Z_BE_torch = torch.from_numpy(Z_BE).float().to(self.device)
        Z_midas_torch = torch.from_numpy(Z_midas_aligned).float().to(self.device)
        confidence_torch = torch.from_numpy(confidence).float().to(self.device)
        edge_weights_torch = torch.from_numpy(edge_weights).float().to(self.device)
        
        # Iterative optimization (Jacobi iterations)
        for iteration in range(self.num_iterations):
            Z_old = Z_fused_torch.clone()
            
            # Data terms
            boundary_term = self.lambda1 * confidence_torch * Z_BE_torch
            mono_term = self.lambda2 * Z_midas_torch
            data_weight = self.lambda1 * confidence_torch + self.lambda2
            
            # Smoothness term (edge-aware bilateral filtering)
            # Average with 4-connected neighbors weighted by edge weights
            Z_padded = F.pad(Z_fused_torch.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode='replicate').squeeze()
            
            Z_up = Z_padded[:-2, 1:-1]
            Z_down = Z_padded[2:, 1:-1]
            Z_left = Z_padded[1:-1, :-2]
            Z_right = Z_padded[1:-1, 2:]
            
            # Edge-aware smoothness
            smoothness_term = self.lambda3 * edge_weights_torch * (Z_up + Z_down + Z_left + Z_right)
            smoothness_weight = 4 * self.lambda3 * edge_weights_torch
            
            # Update (weighted average of all terms)
            Z_fused_torch = (boundary_term + mono_term + smoothness_term) / (data_weight + smoothness_weight + 1e-6)
            
            # Check convergence
            if iteration % 10 == 0 and verbose:
                change = torch.abs(Z_fused_torch - Z_old).mean().item()
                print(f"  Iteration {iteration}: mean change = {change:.6f}")
                
                if change < 1e-5:
                    if verbose:
                        print(f"  Converged at iteration {iteration}")
                    break
        
        Z_fused = Z_fused_torch.cpu().numpy()
        
        if verbose:
            print(f"Final fused depth range: [{Z_fused.min():.2f}, {Z_fused.max():.2f}]")
        
        return Z_fused


def fuse_single_image(Z_BE, Z_midas, confidence, image, 
                      lambda1=10.0, lambda2=1.0, lambda3=0.1,
                      num_iterations=50, device='cuda:0', verbose=False):
    """
    Convenience function to fuse a single depth map
    
    Args:
        Z_BE: Blurry-Edges depth (H, W)
        Z_midas: MiDaS depth (H, W)
        confidence: Confidence map (H, W)
        image: RGB image (H, W, 3)
        lambda1, lambda2, lambda3: Fusion weights
        num_iterations: Number of iterations
        device: torch device
        verbose: Print progress
        
    Returns:
        Z_fused: Fused depth map (H, W)
    """
    fusion = DepthFusion(lambda1=lambda1, lambda2=lambda2, lambda3=lambda3,
                        num_iterations=num_iterations, device=device)
    return fusion.fuse_depths(Z_BE, Z_midas, confidence, image, verbose=verbose)
