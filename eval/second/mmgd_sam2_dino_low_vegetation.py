import os
import sys
import torch
import numpy as np
from tqdm import tqdm
from skimage.io import imread, imsave
from torchvision.ops.boxes import batched_nms
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(os.path.dirname(
    os.path.dirname(current_file_path)))
sys.path.append(parent_directory)

from dynamic_earth import (
    build_mmdet_model,
    extract_bbox_and_cls_from_mmgd,
    bitemporal_match,
    get_model_and_processor,
    instance_ceg)


# Merge change masks into a single binary mask
def merge_masks(change_masks, shape):
    """Merges individual change masks into a single change mask.

    Args:
        change_masks (list of np.array): List of individual change masks.
        shape (tuple): Shape of the output mask (height, width).

    Returns:
        np.array: Merged binary change mask.
    """
    if len(change_masks) == 0:
        return np.zeros((shape[0], shape[1]), dtype=np.uint8)
    
    # Sum the masks and convert to binary (255 for changed areas)
    change_mask = np.sum(change_masks, axis=0).astype(np.uint8)
    change_mask[change_mask > 0] = 255
    
    return change_mask


def filter_predictions(predictions, score_threshold, iou_threshold, device):
    """Filter predictions based on score and NMS.

    Args:
        predictions (dict): Prediction dictionary containing 'bboxes' and 'scores'.
        score_threshold (float): Minimum score to keep.
        iou_threshold (float): IoU threshold for NMS.
        device (str): Device to perform computations on (e.g., 'cuda').

    Returns:
        tuple: Filtered boxes and scores.
    """
    boxes = np.asarray(predictions['bboxes'])
    scores = np.asarray(predictions['scores'])
    
    # Apply score threshold
    keep_indices = np.where(scores >= score_threshold)
    filtered_boxes = boxes[keep_indices]
    filtered_scores = scores[keep_indices]

    # Apply Non-Maximum Suppression (NMS)
    keep_by_nms = batched_nms(
        torch.as_tensor(filtered_boxes, device=device).float(),
        torch.as_tensor(filtered_scores, device=device),
        torch.zeros(filtered_boxes.shape[0], device=device),
        iou_threshold=iou_threshold,
    ).cpu().numpy()

    return filtered_boxes[keep_by_nms], filtered_scores[keep_by_nms]


# Hyperparameters and configurations
TEXT_PROMPT = 'low vegetation. grass' # separated by .
INPUT_DIR = 'data/SECOND2/val'
OUTPUT_DIR = 'output/SECOND_exp/test_mmgd_sam2_dino_low-vegetation'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MMGD_CONFIG = 'dynamic_earth/identifier/mmgd_ext/grounding_dino_swin-l_pretrain_all.py'
MMGD_CHECKPOINT = 'weights/grounding_dino_swin-l_pretrain_all-56d69e78.pth'
MMGD_PRED_SCORE_THRESHOLD = 0.1
MMGD_NMS_IOU_THRESHOLD = 0.7

SAM2_CONFIG = 'configs/sam2.1/sam2.1_hiera_l.yaml'
SAM2_CHECKPOINT = 'weights/sam2.1_hiera_large.pt'

COMPARATOR_MODEL_TYPE = 'DINO'
COMPARATOR_FEATURE_DIM = 768
COMPARATOR_PATCH_SIZE = 16

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Build MM-Grounding-DINO as a identifier
identifier_model = build_mmdet_model(
    model=MMGD_CONFIG,
    weights=MMGD_CHECKPOINT, device=DEVICE)

# build SAM2 image predictor
sam2_model = build_sam2(SAM2_CONFIG, SAM2_CHECKPOINT, device=DEVICE)
sam2_predictor = SAM2ImagePredictor(sam2_model)

# Build DINO as a comparator
comparator_model, comparator_processor = get_model_and_processor(COMPARATOR_MODEL_TYPE, DEVICE)

# Process each image pair in the input directory
for file_name in tqdm(os.listdir(os.path.join(INPUT_DIR, 'im1')), desc="Processing", unit="iteration"):
    # Construct file paths for the images
    img1_path = os.path.join(INPUT_DIR, 'im1', file_name)
    img2_path = os.path.join(INPUT_DIR, 'im2', file_name)

    # Read the input images
    img1 = imread(img1_path)
    img2 = imread(img2_path)

    # Get predictions from the identifier model
    img_predictions = extract_bbox_and_cls_from_mmgd(identifier_model, [img1_path, img2_path],
                                                        texts=TEXT_PROMPT, custom_entities=True)
    img1_predictions, img2_predictions = img_predictions['predictions']
    
    # Filter predictions for both images
    img1_boxes, img1_scores = filter_predictions(img1_predictions, MMGD_PRED_SCORE_THRESHOLD, MMGD_NMS_IOU_THRESHOLD, DEVICE)
    img2_boxes, img2_scores = filter_predictions(img2_predictions, MMGD_PRED_SCORE_THRESHOLD, MMGD_NMS_IOU_THRESHOLD, DEVICE)

    # Init SAM 2 Model and Predict Mask with Box Prompt
    if len(img1_boxes) == 0 and len(img2_boxes) == 0:
        change_mask = np.zeros(img1.shape[:2]).astype(np.uint8)
    elif len(img1_boxes) == 0 or len(img2_boxes) == 0:
        img_boxes = img1_boxes if len(img1_boxes) != 0 else img2_boxes
        img = img1 if len(img1_boxes) != 0 else img2
        
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            sam2_predictor.set_image(img)
            img_masks, img_scores, img_logits = sam2_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=img_boxes,
                mask_input=None,
                multimask_output=False)
        
        # convert the shape to (n, H, W)
        if img_masks.ndim == 4:
            img1_masks = img_masks.squeeze(1)
        
        img_masks = img_masks.astype(np.uint8)
        change_mask = merge_masks(img_masks, img.shape[:2])
    else:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            sam2_predictor.set_image(img1)
            img1_masks, img1_scores, img1_logits = sam2_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=img1_boxes,
                mask_input=None,
                multimask_output=False)
            
            sam2_predictor.set_image(img2)
            img2_masks, img2_scores, img2_logits = sam2_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=img2_boxes,
                mask_input=None,
                multimask_output=False)

        # Post-process the output of the model to get the masks, scores, and logits
        # convert the shape to (n, H, W)
        if img1_masks.ndim == 4:
            img1_masks = img1_masks.squeeze(1)
        if img2_masks.ndim == 4:
            img2_masks = img2_masks.squeeze(1)
        
        img1_mask_num = len(img1_masks)
        
        img_masks = np.concatenate([img1_masks, img2_masks], axis=0)
        cmasks, img1_mask_num = instance_ceg(img_masks, img1_mask_num, iou_threshold=1e-2)
        
        cmasks, img1_mask_num = bitemporal_match(img1, img2, cmasks, comparator_model, comparator_processor,
                                             img1_mask_num, change_confidence_threshold=145, device=DEVICE,
                                             model_config = {'model_type': COMPARATOR_MODEL_TYPE,
                                                             'feature_dim': COMPARATOR_FEATURE_DIM,
                                                             'patch_size': COMPARATOR_PATCH_SIZE})
        # Merge individual change masks into a final change mask
        change_mask = merge_masks(cmasks, img1.shape[:2])
    
    # Save the final change mask to the output directory
    imsave(os.path.join(OUTPUT_DIR, file_name), change_mask)