from typing import Tuple, Union

import numpy as np
import pycocotools.mask as maskUtils
from segment_anything.utils.amg import rle_to_mask, MaskData


def instance_ceg(
    mask_data: Union[MaskData, np.ndarray],
    img1_mask_num: int,
    iou_threshold: float = 1e-2
) -> Tuple[np.ndarray, int]:
    """
    Identify and return the changed instance masks and the updated mask count.

    Parameters:
        mask_data (Union[MaskData, np.ndarray]): Input mask data, which can be either a MaskData object or a NumPy array.
        img1_mask_num (int): The number of masks in the first image.
        iou_threshold (float, optional): IoU threshold for determining instance changes, default is 1e-2.

    Returns:
        Tuple[np.ndarray, int]: An array of changed instances and the updated count of the first image of masks.
    """
    
    # Convert MaskData to a NumPy array if necessary
    if isinstance(mask_data, MaskData):
        mask_data = np.array([rle_to_mask(rle) for rle in mask_data['rles']], dtype=np.uint8)
    elif isinstance(mask_data, np.ndarray):
        mask_data = mask_data.astype(np.uint8)
    
    img1_masks = mask_data[:img1_mask_num]
    img2_masks = mask_data[img1_mask_num:]
    
    # Determine changed instances based on the presence of masks
    if len(img1_masks) == 0:
        change_instances = img2_masks if len(img2_masks) > 0 else np.array([])
        img1_mask_num = 0
    elif len(img2_masks) == 0:
        change_instances = img1_masks
        img1_mask_num = len(img1_masks)
    else:
        img1_rles = [maskUtils.encode(np.asfortranarray(mask)) for mask in img1_masks]
        img2_rles = [maskUtils.encode(np.asfortranarray(mask)) for mask in img2_masks]
        
        ious = maskUtils.iou(img2_rles, img1_rles, [0] * len(img1_rles))
        ious_img1 = ious.sum(axis=0)
        ious_img2 = ious.sum(axis=1)
        
        img1_change_idx = np.where(ious_img1 <= iou_threshold)[0]
        img2_change_idx = np.where(ious_img2 <= iou_threshold)[0]
        
        img1_mask_num = len(img1_change_idx)
        change_instances = np.concatenate(
            [img1_masks[img1_change_idx], img2_masks[img2_change_idx]]
        )
    
    return change_instances, img1_mask_num