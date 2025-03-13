import os
import argparse

import torch
import numpy as np
from skimage.io import imread, imsave
from segment_anything.utils.amg import rle_to_mask
from segment_anything import sam_model_registry, sam_hq_model_registry
from dynamic_earth import (
    identify,
    MaskProposal,
    get_identifier,
    bitemporal_match,
    get_model_and_processor
)


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


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DynamicEarth Demo")
    parser.add_argument("--sam_version", type=str, default="vit_h", 
                        help="SAM ViT version: vit_b / vit_l / vit_h")
    parser.add_argument("--sam_checkpoint", type=str, 
                        default='weights/sam_vit_h_4b8939.pth', 
                        help="Path to SAM checkpoint file")
    parser.add_argument("--sam_hq_checkpoint", type=str, default=None,
                        help="Path to SAM-HQ checkpoint file")
    parser.add_argument("--use_sam_hq", action="store_true", 
                        help="Use SAM-HQ for prediction")
    parser.add_argument('--comparetor_config', type=dict, 
                        default={'model_type': 'DINOv2', 
                                 'feature_dim': 768,
                                 'patch_size': 14})
    parser.add_argument("--input_image_1", type=str, required=True,
                        help="Path to first image file")
    parser.add_argument("--input_image_2", type=str, required=True,
                        help="Path to second image file")
    parser.add_argument("--output_dir", "-o", type=str, default="outputs", 
                        help="Output directory")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run on")
    parser.add_argument("--name_list", type=list,
                        default=['background', 'building'], 
                        help="List of names for identification")
    
    return parser.parse_args()


def load_images(img1_path: str, img2_path: str) -> tuple:
    """
    Load images from the given paths.
    
    Args:
        img1_path (str): Path to the first image file.
        img2_path (str): Path to the second image file.

    Returns:
        tuple: Loaded images (img1, img2).
    """
    img1 = imread(img1_path)
    img2 = imread(img2_path)
    return img1, img2


def initialize_sam(use_sam_hq: bool, sam_version: str, 
                   sam_checkpoint: str, sam_hq_checkpoint: str, 
                   device: str):
    """
    Initialize the SAM model based on the specified parameters.
    
    Args:
        use_sam_hq (bool): Flag to indicate if SAM-HQ should be used.
        sam_version (str): Version of the SAM model.
        sam_checkpoint (str): Path to the SAM checkpoint file.
        sam_hq_checkpoint (str): Path to the SAM-HQ checkpoint file.
        device (str): The device to run the model on.

    Returns:
        model: Initialized SAM model.
    """
    model_registry = sam_hq_model_registry if use_sam_hq else sam_model_registry
    checkpoint = sam_hq_checkpoint if use_sam_hq else sam_checkpoint
    return model_registry[sam_version](checkpoint=checkpoint).to(device)


def setup_mask_proposal(sam) -> MaskProposal:
    """
    Set up the MaskProposal generator with hyperparameters.
    
    Args:
        sam: Initialized SAM model.

    Returns:
        MaskProposal: Configured MaskProposal instance.
    """
    mp = MaskProposal()
    mp.make_mask_generator(
        model=sam,
        points_per_side=32,
        points_per_batch=64,
        pred_iou_thresh=0.5,
        stability_score_thresh=0.95,
        stability_score_offset=0.9,
        box_nms_thresh=0.7,
        min_mask_region_area=0
    )
    mp.set_hyperparameters(
        match_hist=False,
        area_thresh=0.8
    )
    return mp


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    # Load images
    img1, img2 = load_images(args.input_image_1, args.input_image_2)

    # Initialize SAM
    sam = initialize_sam(
        args.use_sam_hq, args.sam_version, args.sam_checkpoint, 
        args.sam_hq_checkpoint, args.device
    )

    # Set up MaskProposal
    mp = setup_mask_proposal(sam)

    # Load models
    comparator_model, comparator_processor = get_model_and_processor(
        args.comparetor_config['model_type'], args.device
    )
    identifier_model, identifier_processor = get_identifier(
        'SegEarth-OV', args.device, name_list=args.name_list
    )

    # Process masks
    masks, img1_mask_num = mp.forward(img1, img2)
    
    # Convert RLE masks to binary numpy arrays
    masks = np.array([rle_to_mask(rle).astype(bool) for rle in masks['rles']])
    
    # Match masks between the two images and get class-agnostic change masks
    cmasks, img1_mask_num = bitemporal_match(img1, img2, masks, comparator_model, comparator_processor,
                                             img1_mask_num, change_confidence_threshold=145, device=args.device,
                                             model_config=args.comparetor_config)
    
    # Identify specific classes of change masks
    cmasks, img1_mask_num = identify(
        img1, img2, cmasks, img1_mask_num, 
        identifier_model, identifier_processor, 
        device=args.device
    )

    # Merge and save the final change mask
    change_mask = merge_masks(cmasks, img1.shape[:2])
    imsave(os.path.join(args.output_dir, os.path.basename(args.input_image_1)), change_mask)

if __name__ == "__main__":
    args = parse_arguments()
    main(args)