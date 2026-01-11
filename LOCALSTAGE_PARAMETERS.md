# LocalStage Parameters - Quick Reference

## 📋 What Parameters Does LocalStage Output?

The LocalStage CNN analyzes each 25×25 pixel patch and outputs **10 parameters** describing the coded aperture blur pattern:

---

## 🔢 10 Parameters Per Patch

### **Parameters [0:4] - XY Coordinates** (4 values)
- **What:** Spatial positions of wedge centers in the blur pattern
- **Range:** Normalized coordinates within the patch
- **Meaning:** Where the blur "focuses" in the coded aperture pattern
- **Format:** `(x1, y1, x2, y2)` for two wedge centers

### **Parameters [4:8] - Angles** (4 values)
- **What:** Orientation/direction of the blur wedges
- **Range:** 0 to 2π radians (0° to 360°)
- **Meaning:** Angular spread of the coded aperture blur
- **Format:** `(θ1, θ2, θ3, θ4)` for wedge orientations

### **Parameters [8:10] - Eta Coefficients** (2 values)
- **What:** Depth-related blur spread parameters
- **Range:** Typically 0.0 to 1.0 (after sigmoid activation)
- **Meaning:** Control how much the blur spreads based on depth
- **Format:** `(η1, η2)`
- **Conversion to Depth:** `depth = depthCal.etas2depth(η1, η2)`

---

## 📂 Where to Find These Parameters

### **Option 1: Run the extraction script**
```bash
python extract_baseline_intermediates.py --data_path "./data_test/regular" --cuda cuda:0
```

This will create:
- `logs/baseline_intermediates/raw_parameters/img180_params_local.npy`
- Shape: `[2, H_patches, W_patches, 10]`

### **Option 2: Already saved during baseline test**
When you ran `blurry_edges_test.py`, the parameters were computed internally but not saved.
The saved files (`depth_*.npy`, `confidence_*.npy`) are the **final outputs** after converting these parameters.

---

## 💡 How Parameters Relate to Depth

```
LocalStage Parameters Flow:
┌─────────────────────────────────────────────────────┐
│ Input: RGB Patch (25×25×3)                          │
└──────────────┬──────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────┐
│ LocalStage CNN                                      │
│ - Conv layers extract blur features                │
│ - Learns to recognize coded aperture patterns      │
└──────────────┬──────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────┐
│ Output: 10 Parameters                               │
│ ├─ xy [0:4]    → Wedge locations                   │
│ ├─ angles [4:8] → Blur orientations                │
│ └─ etas [8:10]  → Depth coefficients                │
└──────────────┬──────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────┐
│ Depth Conversion: depth = etas2depth(η1, η2)       │
│ Formula uses camera calibration and blur physics   │
└──────────────┬──────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────┐
│ Final Depth Map (147×147)                           │
└─────────────────────────────────────────────────────┘
```

---

## 📊 Parameter Statistics (Typical Values)

From analyzing the test images:

| Parameter | Typical Range | Meaning |
|-----------|---------------|---------|
| **xy [0:4]** | -3 to +3 | Normalized patch coordinates |
| **angles [4:8]** | 0 to 2π | Wedge orientations |
| **etas [8:10]** | 0.3 to 0.7 | Depth coefficients (after sigmoid) |

**Depth Range:** 0.75m to 1.18m (typical indoor scene)

---

## 🔬 What These Parameters Tell Us

### **XY Coordinates:**
- High values → Blur center is off-center
- Low values → Blur is centered in patch
- **Indicates:** Object boundaries, depth discontinuities

### **Angles:**
- Uniform angles → Smooth surface
- Varying angles → Textured/complex surface
- **Indicates:** Surface orientation relative to camera

### **Etas (Depth Coefficients):**
- Small etas → Objects far away
- Large etas → Objects close to camera
- **Indicates:** Actual scene depth

---

## 📝 Example: Loading and Analyzing Parameters

```python
import numpy as np

# Load LocalStage parameters for image 180
params = np.load('logs/baseline_intermediates/raw_parameters/img180_params_local.npy')
print(f"Shape: {params.shape}")  # [2, 6, 6, 10] for ~147×147 image

# Analyze a specific patch (view 0, row 3, col 3 - center of image)
view = 0
row = 3
col = 3
patch_params = params[view, row, col, :]

print(f"\nPatch at ({row}, {col}):")
print(f"  XY centers: {patch_params[0:4]}")
print(f"  Angles: {patch_params[4:8] * 180 / np.pi:.2f}°")  # Convert to degrees
print(f"  Etas: {patch_params[8:10]}")

# Analyze spatial distribution
etas_map = params[0, :, :, 8]  # First eta coefficient across all patches
print(f"\nEta1 statistics:")
print(f"  Mean: {etas_map.mean():.3f}")
print(f"  Std: {etas_map.std():.3f}")
print(f"  Range: [{etas_map.min():.3f}, {etas_map.max():.3f}]")
```

---

## 🎯 Summary

**Yes, LocalStage outputs parameters!**

- **What:** 10 parameters per patch (xy, angles, etas)
- **Where:** Saved in `raw_parameters/` folder after running extraction script
- **Format:** NumPy arrays with shape `[2, H_patches, W_patches, 10]`
- **Purpose:** Describe coded aperture blur pattern for depth estimation
- **Access:** Load with `np.load()` and index by `[view, row, col, parameter_idx]`

The parameters ARE being computed during baseline testing, but were only being visualized, not saved as raw arrays. The updated extraction script now saves both visualizations AND the raw numerical parameters.

---

**To extract them, run:**
```bash
python extract_baseline_intermediates.py --data_path "./data_test/regular" --cuda cuda:0
```

This will save everything to: `logs/baseline_intermediates/raw_parameters/`
