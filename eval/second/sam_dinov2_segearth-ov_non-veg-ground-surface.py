import os
import sys
import torch
import numpy as np
from tqdm import tqdm
from skimage.io import imread, imsave
from segment_anything.utils.amg import rle_to_mask
from segment_anything import sam_model_registry, sam_hq_model_registry

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(os.path.dirname(
    os.path.dirname(current_file_path)))
sys.path.append(parent_directory)

from dynamic_earth.sam_ext import MaskProposal
from dynamic_earth.identifier.utils import identify, get_identifier
from dynamic_earth.comparator.bi_match import bitemporal_match
from dynamic_earth.utils import get_model_and_processor


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


# Hyperparameters and configurations
BACKGROUND_CLASS = ['background', 'cropland', 'grass', 'tree', 'forest',
                    'building', 'sports field', 'water']
FOREGROUND_CLASS = ['ground']
TEXT_PROMPT = [','.join(BACKGROUND_CLASS), ','.join(FOREGROUND_CLASS)]
INPUT_DIR = 'data/SECOND2/val'
OUTPUT_DIR = 'output/SECOND_exp/test_sam_dinov2_segearth-ov_non-veg-ground-surface-4'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAM_VERSION = 'vit_h'
SAM_CHECKPOINT = 'weights/sam_vit_h_4b8939.pth'
USE_SAM_HQ = False
SAM_HQ_CHECKPOINT = None

COMPARATOR_MODEL_TYPE = 'DINOv2'
COMPARATOR_FEATURE_DIM = 768
COMPARATOR_PATCH_SIZE = 14

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize SAM model based on configuration
if USE_SAM_HQ:
    sam = sam_hq_model_registry[SAM_VERSION](checkpoint=SAM_HQ_CHECKPOINT).to(DEVICE)
else:
    sam = sam_model_registry[SAM_VERSION](checkpoint=SAM_CHECKPOINT).to(DEVICE)

# Set up the mask proposal generator
mp = MaskProposal()
mp.make_mask_generator(
    model=sam,
    points_per_side=32,
    points_per_batch=64,
    pred_iou_thresh=0.5,
    stability_score_thresh=0.95,
    stability_score_offset=0.9,
    box_nms_thresh=0.7,
    min_mask_region_area=0,
)
mp.set_hyperparameters()

# Build DINOv2 as a comparator
comparator_model, comparator_processor = get_model_and_processor(COMPARATOR_MODEL_TYPE, DEVICE)

# Build SegEarth-OV as a identifier
identifier_model, identifier_processor = get_identifier('SegEarth-OV', DEVICE, name_list=TEXT_PROMPT,
                                                        processor_config={'resize': (448, 448)})

# Process each image pair in the input directory
for file_name in tqdm(os.listdir(os.path.join(INPUT_DIR, 'im1')), desc="Processing", unit="iteration"):
    
    img1_path = os.path.join(INPUT_DIR, 'im1', file_name)
    img2_path = os.path.join(INPUT_DIR, 'im2', file_name)

    # Read the input images
    img1 = imread(img1_path)
    img2 = imread(img2_path)

    # Generate class-agnostic masks using the SAM model
    masks, img1_mask_num = mp.forward(img1, img2)

    # Convert RLE masks to binary numpy arrays
    masks = np.array([rle_to_mask(rle).astype(bool) for rle in masks['rles']])
    
    # Match masks between the two images and get class-agnostic change masks
    cmasks, img1_mask_num = bitemporal_match(img1, img2, masks, comparator_model, comparator_processor,
                                             img1_mask_num, change_confidence_threshold=145, device=DEVICE,
                                             model_config = {'model_type': COMPARATOR_MODEL_TYPE,
                                                             'feature_dim': COMPARATOR_FEATURE_DIM,
                                                             'patch_size': COMPARATOR_PATCH_SIZE})
    
    # Identify specific classes of changes in the masks
    cmasks, img1_mask_num = identify(img1, img2, cmasks, img1_mask_num, identifier_model, identifier_processor,
                                     device=DEVICE, is_instance_class=False)
    
    # Merge individual change masks into a final change mask
    change_mask = merge_masks(cmasks, img1.shape[:2])
    
    # Save the final change mask to the output directory
    imsave(os.path.join(OUTPUT_DIR, file_name), change_mask)