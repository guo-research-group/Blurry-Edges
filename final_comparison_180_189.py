"""
Compare Raw Baseline vs U-Net Densifier for images 180-189
Simple RMSE calculation on the same 10 images
"""
import numpy as np

def compute_rmse(pred, gt):
    """Compute RMSE in meters, convert to cm"""
    valid = gt > 0
    if valid.sum() == 0:
        return 0.0
    diff = pred[valid] - gt[valid]
    rmse = np.sqrt(np.mean(diff ** 2))
    return rmse * 100  # convert to cm

def compute_coverage(depth_map):
    """Calculate percentage of valid depth predictions"""
    return (depth_map > 0).sum() / depth_map.size * 100

# Load ground truth
data_path = './data_test/regular'
depth_maps_all = np.load(f'{data_path}/depth_maps.npy')

print("\n" + "="*80)
print("FINAL COMPARISON: Raw Baseline vs U-Net Densifier (Images 180-189)")
print("="*80 + "\n")

raw_errors = []
threshold_errors = []
raw_coverages = []
threshold_coverages = []

print("Per-image results:")
print("-" * 80)
print(f"{'Image':<8} {'Raw Baseline':<20} {'Threshold':<20}")
print("-" * 80)

for i in range(180, 190):
    # Load depths
    raw_depth = np.load(f'./logs/blurry_edges_depths/raw_depth_{i:03d}.npy')
    threshold_depth = np.load(f'./logs/blurry_edges_depths/depth_{i:03d}.npy')
    gt_depth = depth_maps_all[i]
    
    # Handle dimension mismatch
    if raw_depth.ndim == 3:
        raw_depth = raw_depth.squeeze()
    if threshold_depth.ndim == 3:
        threshold_depth = threshold_depth.squeeze()
    
    # Compute RMSE
    raw_rmse = compute_rmse(raw_depth, gt_depth)
    threshold_rmse = compute_rmse(threshold_depth, gt_depth)
    
    # Compute coverage
    raw_cov = compute_coverage(raw_depth)
    threshold_cov = compute_coverage(threshold_depth)
    
    raw_errors.append(raw_rmse)
    threshold_errors.append(threshold_rmse)
    raw_coverages.append(raw_cov)
    threshold_coverages.append(threshold_cov)
    
    print(f"{i:<8} {raw_rmse:6.2f} cm ({raw_cov:5.1f}%)  {threshold_rmse:6.2f} cm ({threshold_cov:5.1f}%)")

print("-" * 80)
print(f"{'AVERAGE':<8} {np.mean(raw_errors):6.2f} cm ({np.mean(raw_coverages):5.1f}%)  "
      f"{np.mean(threshold_errors):6.2f} cm ({np.mean(threshold_coverages):5.1f}%)")
print("=" * 80)

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"Raw Baseline (100% coverage):      {np.mean(raw_errors):6.2f} ± {np.std(raw_errors):5.2f} cm")
print(f"Threshold Baseline (~24% coverage): {np.mean(threshold_errors):6.2f} ± {np.std(threshold_errors):5.2f} cm")
print("\nU-Net Densifier (from your test):   5.83 cm (100% coverage)")
print("=" * 80)

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)

raw_mean = np.mean(raw_errors)
unet_rmse = 5.83  # From your screenshot

if raw_mean < unet_rmse:
    diff = raw_mean - unet_rmse
    print(f"✓ Raw baseline ({raw_mean:.2f} cm) is {abs(diff):.2f} cm BETTER than U-Net ({unet_rmse} cm)")
    print(f"  → Simply removing the threshold gives better results!")
    print(f"  → The U-Net extension may not provide benefit")
else:
    diff = unet_rmse - raw_mean
    print(f"✓ U-Net ({unet_rmse} cm) is {diff:.2f} cm BETTER than raw baseline ({raw_mean:.2f} cm)")
    print(f"  → The U-Net successfully improves over raw baseline predictions")
    print(f"  → The extension provides clear value!")

print("=" * 80)
