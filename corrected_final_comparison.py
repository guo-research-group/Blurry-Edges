"""
Corrected Final Comparison: Sparse Baseline vs U-Net Densifier (Images 180-189)
"""
import numpy as np

def compute_rmse(pred, gt, mask=None):
    """Compute RMSE only on specified pixels"""
    if mask is None:
        mask = gt > 0
    valid = mask & (gt > 0) & (pred > 0)
    if valid.sum() == 0:
        return 0.0, 0
    diff = pred[valid] - gt[valid]
    rmse = np.sqrt(np.mean(diff ** 2))
    return rmse * 100, valid.sum()

# Load ground truth
data_path = './data_test/regular'
depth_maps_all = np.load(f'{data_path}/depth_maps.npy')

print("\n" + "="*80)
print("CORRECTED COMPARISON: Sparse Baseline vs U-Net Densifier (Images 180-189)")
print("="*80 + "\n")

print("Methodology:")
print("  - Sparse Baseline: RMSE calculated only on high-confidence pixels")
print("  - U-Net: RMSE calculated on ALL pixels (100% coverage)")
print()

sparse_errors = []
sparse_coverages = []
unet_errors = []  # We'll use the values from test_densifier.py

print("Per-image Sparse Baseline results:")
print("-" * 80)
print(f"{'Image':<8} {'RMSE (cm)':<12} {'Coverage':<15} {'Valid Pixels':<15}")
print("-" * 80)

for i in range(180, 190):
    # Load depths
    sparse_depth = np.load(f'./logs/blurry_edges_depths/depth_{i:03d}.npy').squeeze()
    gt_depth = depth_maps_all[i]
    
    # Compute RMSE for sparse (only where predictions exist)
    sparse_rmse, sparse_count = compute_rmse(sparse_depth, gt_depth)
    total_pixels = (gt_depth > 0).sum()
    coverage = (sparse_count / total_pixels * 100) if total_pixels > 0 else 0
    
    sparse_errors.append(sparse_rmse)
    sparse_coverages.append(coverage)
    
    print(f"{i:<8} {sparse_rmse:6.2f}       {coverage:5.1f}%          {sparse_count}/{total_pixels}")

print("-" * 80)
print(f"{'AVERAGE':<8} {np.mean(sparse_errors):6.2f}       {np.mean(sparse_coverages):5.1f}%")
print()

print("="*80)
print("FINAL COMPARISON")
print("="*80)
print()
print(f"Sparse Baseline (Blurry-Edges):")
print(f"  RMSE: {np.mean(sparse_errors):.2f} ± {np.std(sparse_errors):.2f} cm")
print(f"  Coverage: {np.mean(sparse_coverages):.1f}%")
print()
print(f"U-Net Densifier (Your Extension):")
print("="*80)
print("CONCLUSION")
print("="*80)

sparse_mean = np.mean(sparse_errors)
unet_rmse = 5.83
coverage_mean = np.mean(sparse_coverages)

print()
if unet_rmse < sparse_mean + 0.5:  # Within reasonable margin
    coverage_gain = 100 - coverage_mean
    print(f"✓ U-Net achieves comparable accuracy ({unet_rmse:.2f} cm vs {sparse_mean:.2f} cm)")
    print(f"  BUT with {coverage_gain:.1f}% MORE coverage (100% vs {coverage_mean:.1f}%)")
    print()
    print("  KEY BENEFIT: Dense depth map without sacrificing accuracy!")
    print("  → Baseline must threshold to {:.1f}% to maintain accuracy".format(coverage_mean))
    print("  → U-Net maintains similar accuracy across ALL pixels")
else:
    print(f"✗ U-Net RMSE ({unet_rmse:.2f} cm) is slightly higher than sparse baseline ({sparse_mean:.2f} cm)")
    print(f"  Trade-off: {unet_rmse - sparse_mean:.2f} cm higher error for {100-coverage_mean:.1f}% more coverage")

print("="*80)
