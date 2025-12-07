"""
Lightweight Depth Densifier Network
====================================
Learns to densify sparse Blurry-Edges depth maps using learned completion.

Architecture: Shallow U-Net (3 levels) with skip connections
Input: [sparse_depth, boundary_map, confidence_map, RGB_image] (6 channels)
Output: Dense depth map (1 channel)

Total parameters: ~500K (very lightweight for 4GB GPU)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Basic convolutional block with BatchNorm and ReLU"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class DownBlock(nn.Module):
    """Downsample block: 2 conv layers + max pool"""
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        self.conv1 = ConvBlock(in_channels, out_channels)
        self.conv2 = ConvBlock(out_channels, out_channels)
        self.pool = nn.MaxPool2d(2, 2)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        pooled = self.pool(x)
        return pooled, x  # Return both pooled (for next level) and pre-pool (for skip)


class UpBlock(nn.Module):
    """Upsample block: upsample + concat skip + 2 conv layers"""
    def __init__(self, in_channels, skip_channels, out_channels):
        super(UpBlock, self).__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv1 = ConvBlock(in_channels // 2 + skip_channels, out_channels)
        self.conv2 = ConvBlock(out_channels, out_channels)
    
    def forward(self, x, skip):
        x = self.upsample(x)
        # Handle size mismatch due to odd dimensions
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class DepthDensifierUNet(nn.Module):
    """
    Lightweight U-Net for depth completion
    
    Architecture:
        Input: 6 channels (sparse_depth + boundary + confidence + RGB)
        Level 1: 6  -> 32  (147x147 -> 73x73)
        Level 2: 32 -> 64  (73x73 -> 36x36)
        Level 3: 64 -> 128 (36x36 -> 18x18)
        Bottleneck: 128
        Level 3: 128 -> 64  (18x18 -> 36x36)
        Level 2: 64  -> 32  (36x36 -> 73x73)
        Level 1: 32  -> 16  (73x73 -> 147x147)
        Output: 1 channel (dense depth)
    
    Total params: ~500K
    """
    def __init__(self, in_channels=6, out_channels=1):
        super(DepthDensifierUNet, self).__init__()
        
        # Encoder (downsampling path)
        self.down1 = DownBlock(in_channels, 32)      # 6 -> 32
        self.down2 = DownBlock(32, 64)                # 32 -> 64
        self.down3 = DownBlock(64, 128)               # 64 -> 128
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            ConvBlock(128, 256),
            ConvBlock(256, 256)
        )
        
        # Decoder (upsampling path)
        self.up3 = UpBlock(256, 128, 128)             # 256 + 128 -> 128
        self.up2 = UpBlock(128, 64, 64)               # 128 + 64 -> 64
        self.up1 = UpBlock(64, 32, 32)                # 64 + 32 -> 32
        
        # Final output layer
        self.final = nn.Conv2d(32, out_channels, kernel_size=1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor [B, 6, H, W]
                - Channel 0: Sparse depth
                - Channel 1: Boundary map
                - Channel 2: Confidence map
                - Channels 3-5: RGB image
        
        Returns:
            Dense depth map [B, 1, H, W]
        """
        # Encoder
        x1, skip1 = self.down1(x)      # x1: [B, 32, 73, 73], skip1: [B, 32, 147, 147]
        x2, skip2 = self.down2(x1)     # x2: [B, 64, 36, 36], skip2: [B, 64, 73, 73]
        x3, skip3 = self.down3(x2)     # x3: [B, 128, 18, 18], skip3: [B, 128, 36, 36]
        
        # Bottleneck
        x = self.bottleneck(x3)        # [B, 256, 18, 18]
        
        # Decoder with skip connections
        x = self.up3(x, skip3)         # [B, 128, 36, 36]
        x = self.up2(x, skip2)         # [B, 64, 73, 73]
        x = self.up1(x, skip1)         # [B, 32, 147, 147]
        
        # Final output
        dense_depth = self.final(x)    # [B, 1, 147, 147]
        
        return dense_depth
    
    def count_parameters(self):
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class EdgeAwareSmoothnessLoss(nn.Module):
    """
    Edge-aware smoothness loss
    Encourages smooth depth within objects but preserves edges
    """
    def __init__(self):
        super(EdgeAwareSmoothnessLoss, self).__init__()
    
    def forward(self, depth, image):
        """
        Args:
            depth: Predicted depth [B, 1, H, W]
            image: RGB image [B, 3, H, W]
        
        Returns:
            Edge-aware smoothness loss (scalar)
        """
        # Compute depth gradients
        depth_dx = torch.abs(depth[:, :, :, :-1] - depth[:, :, :, 1:])
        depth_dy = torch.abs(depth[:, :, :-1, :] - depth[:, :, 1:, :])
        
        # Compute image gradients (edge detector)
        image_dx = torch.mean(torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:]), dim=1, keepdim=True)
        image_dy = torch.mean(torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :]), dim=1, keepdim=True)
        
        # Edge-aware weights (small gradient at edges, large in flat regions)
        weights_x = torch.exp(-image_dx)
        weights_y = torch.exp(-image_dy)
        
        # Weighted smoothness loss
        smoothness_x = depth_dx * weights_x
        smoothness_y = depth_dy * weights_y
        
        return smoothness_x.mean() + smoothness_y.mean()


class DepthDensifierLoss(nn.Module):
    """
    Combined loss for depth densification
    L_total = λ1 * L1_loss + λ2 * smoothness_loss
    """
    def __init__(self, lambda_l1=1.0, lambda_smooth=0.1):
        super(DepthDensifierLoss, self).__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_smooth = lambda_smooth
        self.l1_loss = nn.L1Loss()
        self.smoothness_loss = EdgeAwareSmoothnessLoss()
    
    def forward(self, pred_depth, gt_depth, image, valid_mask=None):
        """
        Args:
            pred_depth: Predicted dense depth [B, 1, H, W]
            gt_depth: Ground truth depth [B, 1, H, W]
            image: RGB image [B, 3, H, W]
            valid_mask: Valid pixel mask [B, 1, H, W] (optional)
        
        Returns:
            Total loss, L1 loss, smoothness loss
        """
        # L1 loss on valid pixels
        if valid_mask is not None:
            l1 = self.l1_loss(pred_depth * valid_mask, gt_depth * valid_mask)
        else:
            l1 = self.l1_loss(pred_depth, gt_depth)
        
        # Edge-aware smoothness loss
        smooth = self.smoothness_loss(pred_depth, image)
        
        # Total loss
        total = self.lambda_l1 * l1 + self.lambda_smooth * smooth
        
        return total, l1, smooth


if __name__ == "__main__":
    # Test the model
    print("=" * 60)
    print("Testing Depth Densifier U-Net")
    print("=" * 60)
    
    # Create model
    model = DepthDensifierUNet(in_channels=6, out_channels=1)
    
    # Count parameters
    num_params = model.count_parameters()
    print(f"\nTotal trainable parameters: {num_params:,} ({num_params/1e6:.2f}M)")
    
    # Test forward pass
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Create dummy input
    batch_size = 2
    x = torch.randn(batch_size, 6, 147, 147).to(device)
    
    print(f"\nInput shape: {x.shape}")
    
    # Forward pass
    with torch.no_grad():
        output = model(x)
    
    print(f"Output shape: {output.shape}")
    
    # Test loss
    loss_fn = DepthDensifierLoss(lambda_l1=1.0, lambda_smooth=0.1)
    gt_depth = torch.randn(batch_size, 1, 147, 147).to(device)
    image = x[:, 3:6, :, :]  # RGB channels
    
    total_loss, l1_loss, smooth_loss = loss_fn(output, gt_depth, image)
    print(f"\nLoss test:")
    print(f"  Total loss: {total_loss.item():.4f}")
    print(f"  L1 loss: {l1_loss.item():.4f}")
    print(f"  Smoothness loss: {smooth_loss.item():.4f}")
    
    print("\n" + "=" * 60)
    print("✓ Model test passed!")
    print("=" * 60)
