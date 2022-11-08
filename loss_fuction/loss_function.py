import segmentation_models_pytorch as smp
import torch
from sklearn.metrics import confusion_matrix


JaccardLoss = smp.losses.JaccardLoss(mode='multilabel')
DiceLoss    = smp.losses.DiceLoss(mode='multilabel')
LovaszLoss  = smp.losses.LovaszLoss(mode='multilabel', per_image=False)


def dice_coef(y_true, y_pred, thr=0.5, dim=(2,3), epsilon=0.001, alpha=0.1):
    y_true = y_true.to(torch.float32)
    y_pred = (y_pred>thr).to(torch.float32)
    inter = (y_true*y_pred).sum(dim=dim)
    den = y_true.sum(dim=dim) + y_pred.sum(dim=dim)
    dice = ((2*inter+epsilon)/(den+epsilon)).mean(dim=(1,0))
    return dice


def iou_coef(y_true, y_pred, thr=0.5, dim=(2, 3), epsilon=0.001):
    y_true = y_true.to(torch.float32)
    y_pred = (y_pred>thr).to(torch.float32)
    inter = (y_true*y_pred).sum(dim=dim)
    union = (y_true + y_pred - y_true*y_pred).sum(dim=dim)
    iou = ((inter+epsilon)/(union+epsilon)).mean(dim=(1,0))
    return iou


def criterion(y_pred, y_true, CFG):
    if CFG['attention'] == 'scse':
        BCELoss = smp.losses.SoftBCEWithLogitsLoss(smooth_factor=0.1)
        TverskyLoss = smp.losses.TverskyLoss(mode='multilabel', smooth=0.1, log_loss=False)
        return 0.5 * BCELoss(y_pred, y_true) + 0.5 * TverskyLoss(y_pred, y_true)
    else:
        BCELoss = smp.losses.SoftBCEWithLogitsLoss()
        TverskyLoss = smp.losses.TverskyLoss(mode='multilabel', log_loss=False)
        return 0.5 * BCELoss(y_pred, y_true) + 0.5 * TverskyLoss(y_pred, y_true)