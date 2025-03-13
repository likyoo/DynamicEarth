import os
import sys
import torch
import numpy as np
from tqdm import tqdm
from skimage.io import imread, imsave

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(os.path.dirname(
    os.path.dirname(current_file_path)))
sys.path.append(parent_directory)

from dynamic_earth.utils import get_model_and_processor
from dynamic_earth.comparator.ins_ceg import instance_ceg
from dynamic_earth.comparator.bi_match import bitemporal_match
from dynamic_earth.identifier.ape_ext import build_ape, extract_prediction_from_ape

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


def extract_instance_data(predictions):
    """Extract boxes, masks, and scores from the predictions.

    Args:
        predictions (dict): Predictions from the model containing instances.

    Returns:
        tuple: A tuple containing boxes, masks, and scores as numpy arrays.
    """
    boxes = predictions['instances'].pred_boxes.tensor.cpu().numpy()
    masks = predictions['instances'].pred_masks.cpu().numpy()
    scores = predictions['instances'].scores.cpu().numpy()
    return boxes, masks, scores


# Hyperparameters and configurations
TEXT_PROMPT = 'tree' # separated by commas
INPUT_DIR = 'data/SECOND2/val'
OUTPUT_DIR = 'output/SECOND_exp/test_ape_dinov2_tree'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

APE_CHECKPOINT = 'third_party/APE/model_final.pth'
APE_CONFIG_FILE = 'third_party/APE/configs/LVISCOCOCOCOSTUFF_O365_OID_VGR_SA1B_REFCOCO_GQA_PhraseCut_Flickr30k/ape_deta/ape_deta_vitl_eva02_clip_vlf_lsj1024_cp_16x4_1080k.py'
APE_CONFIDENCE_THRESHOLD = 0.2
APE_OPT = [
    f'train.init_checkpoint={APE_CHECKPOINT}',  # Path to the initial checkpoint
    'model.model_vision.select_box_nums_for_evaluation=500',  # Number of boxes for evaluation
    'model.model_vision.text_feature_bank_reset=True',  # Reset feature bank
    'model.model_vision.backbone.net.xattn=True',  # Use xformers
]
# f'train.device={device}' # Error: Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.

COMPARATOR_MODEL_TYPE = 'DINOv2'
COMPARATOR_FEATURE_DIM = 768
COMPARATOR_PATCH_SIZE = 14

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Build APE as a identifier
identifier_model = build_ape(APE_CONFIG_FILE, APE_CONFIDENCE_THRESHOLD, APE_OPT)

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

    # Extract predictions from APE model
    img1_predictions = extract_prediction_from_ape(identifier_model, img1_path, TEXT_PROMPT)
    img2_predictions = extract_prediction_from_ape(identifier_model, img2_path, TEXT_PROMPT)

    # Retrieve masks and scores for the bi-temporal images
    img1_boxes, img1_masks, img1_scores = extract_instance_data(img1_predictions)
    img2_boxes, img2_masks, img2_scores = extract_instance_data(img2_predictions)

    # Combine masks from both images
    img_masks = np.concatenate([img1_masks, img2_masks], axis=0)
    img1_mask_num = len(img1_masks)

    # Filter change masks by comparing instance masks
    cmasks, img1_mask_num = instance_ceg(img_masks, img1_mask_num, iou_threshold=1e-2)

    # Filter change masks by comparing instance features
    cmasks, img1_mask_num = bitemporal_match(img1, img2, cmasks, comparator_model, comparator_processor,
                                             img1_mask_num, change_confidence_threshold=155, device=DEVICE,
                                             model_config = {'model_type': COMPARATOR_MODEL_TYPE,
                                                             'feature_dim': COMPARATOR_FEATURE_DIM,
                                                             'patch_size': COMPARATOR_PATCH_SIZE})

    # Merge individual change masks into a final change mask
    change_mask = merge_masks(cmasks, img1.shape[:2])
    
    # Save the final change mask to the output directory
    imsave(os.path.join(OUTPUT_DIR, file_name), change_mask)