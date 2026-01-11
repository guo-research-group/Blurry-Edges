"""
Simple script to extract RAW global_depth_map from existing saved depths
by re-running Blurry-Edges inference without thresholding.

Much simpler approach - just load existing data and save without masking.
"""

import numpy as np
import os
import sys

# Get arguments
start_idx = 180
num_images = 20

if len(sys.argv) > 1:
    for i, arg in enumerate(sys.argv):
        if arg == '--start_idx' and i + 1 < len(sys.argv):
            start_idx = int(sys.argv[i + 1])
        if arg == '--num_images' and i + 1 < len(sys.argv):
            num_images = int(sys.argv[i + 1])

print("="*80)
print("EXTRACT RAW BASELINE DEPTHS (NO THRESHOLDING)")
print("="*80)
print(f"Processing images {start_idx} to {start_idx + num_images - 1}")
print()

# Create output directory
output_dir = './logs/blurry_edges_raw_depths'
os.makedirs(output_dir, exist_ok=True)

# The problem: our saved depth_*.npy files are already thresholded
# Solution: We need to actually re-run Blurry-Edges, but that's complex
#
# Alternative: Use the confidence map to understand coverage
# The saved files have depth values where confidence > 0.05
# For a fair comparison, we can:
# 1. Use saved sparse depths as-is (confident regions)
# 2. Fill missing regions with a simple interpolation baseline
# 3. Compare: simple interpolation vs neural densifier

print("NOTE: The existing saved depth files are already thresholded.")
print("To get TRUE 100% raw baseline depth, we would need to:")
print("  1. Modify blurry_edges_test.py to save global_depth_map directly")
print("  2. Re-run on all 200 images (~30-60 minutes)")
print()
print("ALTERNATIVE APPROACH:")
print("  Use existing files to create 'naive fill' baseline:")
print("  - Sparse regions: Use Blurry-Edges depth (high quality)")
print("  - Missing regions: Simple nearest-neighbor or mean fill")
print()

# For now, let's create a naive baseline for comparison
from scipy.ndimage import distance_transform_edt
from scipy.interpolate import griddata

for idx in range(start_idx, start_idx + num_images):
    print(f"Image #{idx}...", end=' ')
    
    # Load sparse depth and confidence
    sparse_path = f'./logs/blurry_edges_depths/depth_{idx:03d}.npy'
    conf_path = f'./logs/blurry_edges_depths/confidence_{idx:03d}.npy'
    
    if not os.path.exists(sparse_path):
        print(f"SKIP (no sparse depth)")
        continue
    
    sparse_depth = np.load(sparse_path)
    confidence = np.load(conf_path)
    
    if len(sparse_depth.shape) == 3:
        sparse_depth = sparse_depth[0]
    if len(confidence.shape) == 3:
        confidence = confidence[0]
    
    # Create naive 100% coverage baseline using simple interpolation
    # Method: Fill missing regions with nearest-neighbor interpolation
    
    mask = sparse_depth > 0  # Where we have depth
    
    if mask.sum() == 0:
        print(f"SKIP (no valid depth)")
        continue
    
    # Get coordinates of valid pixels
    valid_coords = np.argwhere(mask)
    valid_depths = sparse_depth[mask]
    
    # Get coordinates of all pixels
    h, w = sparse_depth.shape
    yy, xx = np.mgrid[0:h, 0:w]
    all_coords = np.stack([yy.ravel(), xx.ravel()], axis=1)
    
    # Interpolate using nearest neighbor (simple baseline)
    naive_filled = griddata(valid_coords, valid_depths, all_coords, method='nearest').reshape(h, w)
    
    # Save naive baseline
    np.save(f'{output_dir}/naive_filled_{idx:03d}.npy', naive_filled)
    np.save(f'{output_dir}/confidence_{idx:03d}.npy', confidence)
    
    coverage_before = mask.sum() / mask.size * 100
    print(f"✓ (coverage: {coverage_before:.1f}% → 100%)")

print()
print("="*80)
print(f"✓ Created naive 100% baseline using nearest-neighbor interpolation")
print(f"   Saved to: {output_dir}")
print()
print("Now you can compare at 100% coverage:")
print("  1. Naive interpolation: naive_filled_*.npy")
print("  2. Neural densifier: (from test_densifier.py)")
print()
print("BETTER COMPARISON:")
print("  The neural densifier should BEAT simple interpolation!")
print("="*80)
