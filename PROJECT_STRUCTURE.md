# Blurry-Edges Extension: Depth Densification

## 📁 Repository Structure

### **Baseline (Original Blurry-Edges)**
- `blurry_edges_test.py` - Main test script for baseline
- `blurry_edges_test_big.py` - Test script for larger images
- `global_training.py` - Global stage training
- `local_training.py` - Local stage training
- `global_data_pre_cal.py` - Data preprocessing
- `test_data_generator.py` - Test data generator
- `train_val_data_generator.py` - Training/validation data generator
- `models/global_stage.py` - Global stage architecture
- `models/local_stage.py` - Local stage architecture
- `models/depth_completion_unet.py` - Depth completion module
- `pretrained_weights/pretrained_*.pth` - Baseline pretrained weights

### **Extension (Depth Densifier)**
- `models/depth_densifier.py` - **Our U-Net densifier architecture**
- `train_densifier.py` - **Training script for densifier**
- `test_densifier.py` - **Evaluation script on held-out test set**
- `pretrained_weights/best_densifier.pth` - **Trained densifier weights**

### **Analysis & Validation**
- `test_threshold_comparison.py` - **Validates extension vs threshold lowering**
- `analyze_regional_quality.py` - **Regional quality analysis (confident vs missing pixels)**
- `threshold_comparison_results.txt` - Threshold comparison results
- `quality_analysis_results.txt` - Regional quality analysis results

### **Visualization**
- `visualize_densifier.py` - Comprehensive 12-panel visualization
- `visualize_simple.py` - Simple 6-panel before/after comparison

### **Data & Outputs**
- `data_test/` - Test dataset (200 images)
- `logs/blurry_edges_depths/` - Saved sparse depth maps from baseline
- `pic/` - Sample images

---

## 🎯 Extension Summary

**Problem:** Blurry-Edges produces sparse depth (only 24% coverage)

**Solution:** Lightweight U-Net densifier that fills missing regions

**Results:**
- Coverage: 24% → 100% (+311%)
- RMSE on confident pixels: 5.67 → 5.56 cm (-2% better!)
- RMSE on missing pixels: N/A → 6.17 cm (filled!)
- Overall RMSE: 6.08 cm (100% coverage)

---

## 🚀 Quick Start

### 1. Test Baseline
```bash
python blurry_edges_test.py
```

### 2. Evaluate Densifier (on held-out test set 180-199)
```bash
python test_densifier.py --start_idx 180 --num_images 20 --cuda cuda:0
```

### 3. Validate Extension
```bash
python test_threshold_comparison.py --start_idx 180 --num_images 20 --cuda cuda:0
python analyze_regional_quality.py
```

### 4. Generate Visualizations
```bash
# Simple before/after comparison
python visualize_simple.py --start_idx 180 --num_images 5 --cuda cuda:0

# Comprehensive analysis
python visualize_densifier.py --start_idx 180 --num_images 5 --cuda cuda:0
```

---

## 📊 Key Results

| Method | Coverage | RMSE (confident) | RMSE (overall) |
|--------|----------|------------------|----------------|
| Sparse Baseline | 24% | 5.67 cm | N/A |
| **Densifier** | **100%** | **5.56 cm** | **6.08 cm** |

**Validation:** Neural densifier is 2% better on confident pixels while filling 76% missing regions!

---

## 📝 Data Split (No Leakage)

- **Training:** Images 0-139 (140 images, 70%)
- **Validation:** Images 140-179 (40 images, 20%)
- **Test (Held-out):** Images 180-199 (20 images, 10%)

Test set was NEVER used during training or validation.

---

## 🏆 Why This Extension Works

1. **Maintains Quality:** 5.56 cm on confident pixels (vs 5.67 cm sparse)
2. **Fills Gaps:** 6.17 cm on missing pixels (vs infinite error)
3. **Complete Coverage:** 100% depth everywhere
4. **Learned Intelligence:** Uses context, edges, and RGB to fill smartly

---

## 📄 Files Kept

- ✅ All baseline code
- ✅ Extension training/testing scripts
- ✅ Validation and comparison scripts
- ✅ Visualization scripts
- ✅ Results and analysis outputs
- ❌ Debug files removed
- ❌ Temporary files removed
- ❌ Failed experiments removed
