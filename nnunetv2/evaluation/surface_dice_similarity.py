import SimpleITK as sitk
import numpy as np
from surface_distance import metrics

def create_region_from_mask(mask, join_labels: tuple):
    mask_new = np.zeros_like(mask, dtype=np.uint8)
    for l in join_labels:
        mask_new[mask == l] = 1
    return mask_new

def evaluate_case_sdc(file_pred: str, file_gt: str):
    image_gt = sitk.GetArrayFromImage(sitk.ReadImage(file_gt))
    image_pred = sitk.GetArrayFromImage(sitk.ReadImage(file_pred))
    results = []
    for r in regions:
        mask_pred = create_region_from_mask(image_pred, r).astype(bool)
        mask_gt = create_region_from_mask(image_gt, r).astype(bool)
        if np.sum(mask_gt) == 0 and np.sum(mask_pred) == 0:
            sdc = np.nan  
        else:
            surface_distances = metrics.compute_surface_distances(
                                        mask_gt, mask_pred, [1,1,1]
                                    )
            sdc =  metrics.compute_surface_dice_at_tolerance(
                                        surface_distances, 
                                        tolerance_mm=1)
        results.append(sdc)
    return results