"""
Monocular Depth Estimation using MiDaS
Generates dense depth maps from single images (using one of the defocused images)
"""
import os
import torch
import numpy as np
from data import TestDataset
import argparse

def load_midas_model(device):
    """Load pretrained MiDaS model (using smaller MiDaS_small for faster download)"""
    print("Loading MiDaS model (MiDaS_small - 105MB)...")
    # Download and load pretrained model - using smaller model (105MB vs 1.3GB)
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
    midas.to(device)
    midas.eval()
    
    # Load transforms
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
    transform = midas_transforms.small_transform
    
    print("MiDaS model loaded successfully!")
    return midas, transform

def run_midas_inference(midas_model, transform, image, device):
    """
    Run MiDaS inference on a single image
    
    Args:
        midas_model: Loaded MiDaS model
        transform: MiDaS transform function
        image: Input image (H, W, 3) in range [0, 1]
        device: torch device
        
    Returns:
        depth: Depth map (H, W) - disparity map (inverse depth)
    """
    # Convert to 0-255 uint8 format expected by MiDaS
    img_uint8 = (image * 255).astype(np.uint8)
    
    # Apply MiDaS transforms
    input_batch = transform(img_uint8).to(device)
    
    # Inference
    with torch.no_grad():
        prediction = midas_model(input_batch)
        
        # Resize to original resolution
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=image.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    
    depth = prediction.cpu().numpy()
    return depth

def main():
    parser = argparse.ArgumentParser(description='Run MiDaS monocular depth estimation')
    parser.add_argument('--data_path', type=str, default='./data_test/regular',
                        help='Path to test data directory')
    parser.add_argument('--cuda', type=str, default='cuda:0',
                        help='CUDA device')
    parser.add_argument('--output_path', type=str, default='./midas_predictions',
                        help='Path to save MiDaS predictions')
    parser.add_argument('--num_images', type=int, default=10,
                        help='Number of images to process')
    args = parser.parse_args()
    
    # Setup device
    device = torch.device(args.cuda if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_path, exist_ok=True)
    
    # Load MiDaS model
    midas_model, transform = load_midas_model(device)
    
    # Load test dataset
    print(f"\nLoading test dataset from {args.data_path}")
    test_dataset = TestDataset(device=device, data_path=args.data_path)
    print(f"Loaded {len(test_dataset)} image pairs")
    
    # Process images
    print(f"\n{'='*60}")
    print(f"RUNNING MIDAS MONOCULAR DEPTH ESTIMATION")
    print(f"{'='*60}\n")
    
    midas_depths = []
    
    for i in range(min(args.num_images, len(test_dataset))):
        print(f"Processing image {i+1}/{args.num_images}...")
        
        # Get image pair (use first image of the pair for monocular depth)
        img_ny, depth_gt = test_dataset[i]
        
        # img_ny shape: [2, H, W, 3] - two defocused images
        # Use the first image for monocular depth estimation
        img_for_midas = img_ny[0].cpu().numpy()  # [H, W, 3]
        
        # Run MiDaS inference
        midas_depth = run_midas_inference(midas_model, transform, img_for_midas, device)
        midas_depths.append(midas_depth)
        
        print(f"  MiDaS depth range: [{midas_depth.min():.2f}, {midas_depth.max():.2f}]")
    
    # Save predictions
    midas_depths = np.array(midas_depths)
    output_file = os.path.join(args.output_path, 'midas_depths.npy')
    np.save(output_file, midas_depths)
    
    print(f"\n{'='*60}")
    print(f"Saved MiDaS predictions: {output_file}")
    print(f"Shape: {midas_depths.shape}")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
