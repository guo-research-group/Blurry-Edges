"""
Calculate RMSE for first 4 images (000-003) where we have raw baseline depths
Compare: Raw Baseline vs Thresholded Baseline
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
print("RMSE COMPARISON FOR FIRST 4 IMAGES (000-003)")
print("="*80 + "\n")

raw_errors = []
threshold_errors = []
raw_coverages = []
threshold_coverages = []

for i in range(4):
    # Load depths
    raw_depth = np.load(f'./logs/blurry_edges_depths/raw_depth_{i:03d}.npy')
    threshold_depth = np.load(f'./logs/blurry_edges_depths/depth_{i:03d}.npy')
    gt_depth = depth_maps_all[i]
    
    # Handle dimension mismatch (squeeze if needed)
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
    
    print(f"Image {i:03d}:")
    print(f"  Raw Baseline:      {raw_rmse:6.2f} cm  (coverage: {raw_cov:5.1f}%)")
    print(f"  Threshold (>0.05): {threshold_rmse:6.2f} cm  (coverage: {threshold_cov:5.1f}%)")
    print()

print("="*80)
print("AVERAGE RESULTS")
print("="*80)
print(f"Raw Baseline:      {np.mean(raw_errors):6.2f} ± {np.std(raw_errors):5.2f} cm  (coverage: {np.mean(raw_coverages):5.1f}%)")
print(f"Threshold (>0.05): {np.mean(threshold_errors):6.2f} ± {np.std(threshold_errors):5.2f} cm  (coverage: {np.mean(threshold_coverages):5.1f}%)")
print("="*80)

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)

raw_mean = np.mean(raw_errors)
threshold_mean = np.mean(threshold_errors)
raw_cov_mean = np.mean(raw_coverages)
threshold_cov_mean = np.mean(threshold_coverages)

if raw_mean < threshold_mean:
    print(f"✓ Raw baseline is {threshold_mean - raw_mean:.2f} cm BETTER than thresholded!")
    print(f"  AND has {raw_cov_mean - threshold_cov_mean:.1f}% MORE coverage")
    print("\n  → Your intuition was CORRECT!")
    print("  → Simply removing the threshold gives better results")
    print("  → The U-Net extension may not be necessary")
else:
    print(f"✗ Raw baseline is {raw_mean - threshold_mean:.2f} cm WORSE than thresholded")
    print(f"  But has {raw_cov_mean - threshold_cov_mean:.1f}% MORE coverage")

print("="*80)
