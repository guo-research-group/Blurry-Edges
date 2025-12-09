import numpy as np

def eval_depth(pred, gt, msk, crop=0, tau_n=1.25, z_min=0.75, z_max=1.18):
    pred = pred.clip(z_min, z_max)
    if crop > 0:
        pred = pred[:, crop:-crop, crop:-crop]
        gt = gt[:, crop:-crop, crop:-crop]
        msk = msk[:, crop:-crop, crop:-crop]
    error = np.abs(gt - pred)
    pred_norm = ((pred - z_min) / (z_max - z_min)).clip(0,1)
    gt_norm = ((gt - z_min) / (z_max - z_min)).clip(0,1)
    msk_num = np.sum(msk)
    gt_pred = gt_norm/(pred_norm+1e-8)
    pred_gt = pred_norm/(gt_norm+1e-8)
    acc = np.maximum(gt_pred, pred_gt)
    delta1 = np.sum((acc < tau_n) * msk) / msk_num
    delta2 = np.sum((acc < tau_n**2) * msk) / msk_num    
    delta3 = np.sum((acc < tau_n**3) * msk) / msk_num
    RMSE = np.sqrt(np.sum(error ** 2 * msk) / msk_num)
    AbsRel = np.sum(error * msk / gt * msk) / msk_num
    return delta1, delta2, delta3, RMSE*100, AbsRel*100

def compute_errors(gt, pred, tau_n=1.25):
    """
    Compute depth estimation errors for fusion evaluation
    Compatible with test_fusion.py
    
    Args:
        gt: Ground truth depth (flattened or 2D array)
        pred: Predicted depth (flattened or 2D array)
        tau_n: Threshold for delta metric
        
    Returns:
        Dictionary with delta1, rmse, abs_rel metrics
    """
    gt = np.array(gt).flatten()
    pred = np.array(pred).flatten()
    
    # Filter valid pixels
    valid = (gt > 0) & (pred > 0) & np.isfinite(gt) & np.isfinite(pred)
    gt = gt[valid]
    pred = pred[valid]
    
    if len(gt) == 0:
        return {'delta1': 0.0, 'rmse': 999.0, 'abs_rel': 999.0}
    
    # Clip to reasonable range
    z_min, z_max = gt.min(), gt.max()
    pred = np.clip(pred, z_min - 0.5, z_max + 0.5)
    
    # Normalize for delta metric
    pred_norm = (pred - z_min) / (z_max - z_min + 1e-8)
    gt_norm = (gt - z_min) / (z_max - z_min + 1e-8)
    
    # Delta metric (percentage of pixels within threshold)
    ratio = np.maximum(gt_norm / (pred_norm + 1e-8), pred_norm / (gt_norm + 1e-8))
    delta1 = np.mean(ratio < tau_n)
    
    # RMSE (in cm)
    rmse = np.sqrt(np.mean((gt - pred) ** 2)) * 100
    
    # Absolute Relative Error (in cm)
    abs_rel = np.mean(np.abs(gt - pred) / (gt + 1e-8)) * 100
    
    return {
        'delta1': float(delta1),
        'rmse': float(rmse),
        'abs_rel': float(abs_rel)
    }