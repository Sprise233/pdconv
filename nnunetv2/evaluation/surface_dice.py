import numpy as np
from scipy.ndimage import distance_transform_edt


def calculate_sdc(pred, gt, tau):
    """
    Calculate Surface Dice Coefficient (SDC).

    Args:
        pred (np.array): Predicted binary segmentation (0 or 1).
        gt (np.array): Ground truth binary segmentation (0 or 1).
        tau (float): Distance threshold.

    Returns:
        float: Surface Dice Coefficient.
    """
    # Get the surface (boundary) of the segmentation
    pred_surface = np.logical_xor(pred, distance_transform_edt(~pred) <= 1)
    gt_surface = np.logical_xor(gt, distance_transform_edt(~gt) <= 1)

    # Compute distances from surfaces
    dist_pred_to_gt = distance_transform_edt(~gt)
    dist_gt_to_pred = distance_transform_edt(~pred)

    # Match surfaces within the threshold tau
    pred_close_to_gt = (dist_pred_to_gt[pred_surface] <= tau).sum()
    gt_close_to_pred = (dist_gt_to_pred[gt_surface] <= tau).sum()

    # Total surface points
    total_surface = pred_surface.sum() + gt_surface.sum()

    if total_surface == 0:  # Handle edge case
        return 1.0 if pred.sum() == gt.sum() == 0 else 0.0

    return (pred_close_to_gt + gt_close_to_pred) / total_surface


# Example usage
pred = np.array([[0, 1, 1, 0],
                 [0, 1, 1, 0],
                 [0, 0, 0, 0]])

gt = np.array([[0, 1, 0, 0],
               [0, 1, 1, 0],
               [0, 0, 1, 0]])

tau = 1.5
sdc = calculate_sdc(pred, gt, tau)
print(f"Surface Dice Coefficient: {sdc:.4f}")
