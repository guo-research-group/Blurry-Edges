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