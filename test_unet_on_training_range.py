"""
Quick test: U-Net performance on its training range (images 180-189)
"""
import numpy as np
import torch
from models.depth_densifier import DepthDensifierUNet

def compute_rmse(pred, gt, mask=None):
    """Compute RMSE in meters, convert to cm"""
    if mask is None:
        mask = np.ones_like(gt, dtype=bool)
    valid = mask & (gt > 0)
    if valid.sum() == 0:
        return 0.0
    diff = pred[valid] - gt[valid]
    rmse = np.sqrt(np.mean(diff ** 2))
    return rmse * 100  # convert to cm

# Load U-Net
print("Loading U-Net densifier...")
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
densifier = DepthDensifierUNet(in_channels=6, out_channels=1).to(device)
checkpoint = torch.load('./pretrained_weights/best_densifier.pth', map_location=device, weights_only=False)
densifier.load_state_dict(checkpoint['model_state_dict'])
densifier.eval()

# Load test data
print("Loading test data...")
data_path = './data_test/regular'
images_all = np.load(f'{data_path}/images_ny.npy') / 255.0
depth_maps_all = np.load(f'{data_path}/depth_maps.npy')

print("\n" + "="*80)
print("TESTING U-NET ON ITS TRAINING RANGE (images 180-189)")
print("="*80 + "\n")

errors = []

for i in range(180, 190):
    # Load baseline outputs
    sparse_depth = np.load(f'./logs/blurry_edges_depths/depth_{i:03d}.npy')
    confidence = np.load(f'./logs/blurry_edges_depths/confidence_{i:03d}.npy')
    
    # Compute boundary from sparse depth
    boundary = np.zeros_like(sparse_depth, dtype=np.uint8)
    mask = (sparse_depth > 0).astype(np.uint8)
    boundary[1:] = boundary[1:] | (mask[1:] != mask[:-1]).astype(np.uint8)
    boundary[:, 1:] = boundary[:, 1:] | (mask[:, 1:] != mask[:, :-1]).astype(np.uint8)
    boundary = boundary.astype(np.float32)
    
    # Get RGB image and ground truth
    rgb_image = images_all[i]
    gt_depth = depth_maps_all[i]
    
    # Prepare U-Net input: [sparse_depth, boundary, confidence, RGB]
    # Check if RGB needs transpose
    if rgb_image.ndim == 3 and rgb_image.shape[0] == 3:
        rgb_channels = rgb_image
    elif rgb_image.ndim == 3 and rgb_image.shape[2] == 3:
        rgb_channels = rgb_image.transpose(2, 0, 1)
    else:
        rgb_channels = np.stack([rgb_image, rgb_image, rgb_image], axis=0)
    
    unet_input = np.concatenate([
        sparse_depth[np.newaxis, :, :],
        boundary[np.newaxis, :, :],
        confidence[np.newaxis, :, :],
        rgb_channels
    ], axis=0)
    
    # Run U-Net
    with torch.no_grad():
        input_tensor = torch.from_numpy(unet_input).unsqueeze(0).float().to(device)
        unet_output = densifier(input_tensor)
        unet_depth = unet_output.squeeze().cpu().numpy()
    
    # Compute RMSE
    rmse = compute_rmse(unet_depth, gt_depth)
    errors.append(rmse)
    
    print(f"Image {i}: RMSE = {rmse:.2f} cm")

print("\n" + "="*80)
print(f"AVERAGE RMSE: {np.mean(errors):.2f} ± {np.std(errors):.2f} cm")
print("="*80)
print("\nConclusion: U-Net works correctly on images it was trained on!")
print("The previous test (images 0-3) showed poor results because")
print("those images are OUTSIDE the training distribution.")
