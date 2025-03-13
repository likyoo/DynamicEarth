import os
import argparse

import torch
import numpy as np
from skimage.io import imread, imsave

from dynamic_earth.utils import get_model_and_processor
from dynamic_earth.comparator.ins_ceg import instance_ceg
from dynamic_earth.comparator.bi_match import bitemporal_match
from dynamic_earth.identifier.ape_ext import build_ape, extract_prediction_from_ape

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


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="DynamicEarth Demo")
    parser.add_argument("--input_image_1", type=str, required=True,
                        help="Path to directory containing first set of images")
    parser.add_argument("--input_image_2", type=str, required=True,
                        help="Path to directory containing second set of images")
    parser.add_argument("--output_dir", "-o", type=str, default="outputs",
                        help="Output directory for change masks")
    parser.add_argument("--text_prompt", type=str, default="building,house",
                        help="Comma-separated text prompts for identification")
    parser.add_argument("--ape_config", type=str,
                        default="third_party/APE/configs/LVISCOCOCOCOSTUFF_O365_OID_VGR_SA1B_REFCOCO_GQA_PhraseCut_Flickr30k/ape_deta/ape_deta_vitl_eva02_clip_vlf_lsj1024_cp_16x4_1080k.py",
                        help="Path to APE config file")
    parser.add_argument("--ape_checkpoint", type=str,
                        default="third_party/APE/model_final.pth",
                        help="Path to APE checkpoint file. " \
                            "Download from https://huggingface.co/shenyunhang/APE/blob/main/configs/LVISCOCOCOCOSTUFF_O365_OID_VGR_SA1B_REFCOCO_GQA_PhraseCut_Flickr30k/ape_deta/ape_deta_vitl_eva02_clip_vlf_lsj1024_cp_16x4_1080k_mdl_20230829_162438/model_final.pth")
    parser.add_argument("--confidence_thresh", type=float, default=0.2,
                        help="Confidence threshold for predictions")
    parser.add_argument("--comparetor_config", type=dict,
                        default={'model_type': 'DINO',
                                 'feature_dim': 768,
                                 'patch_size': 16})
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run on")
    return parser.parse_args()


def initialize_models(args):
    """Initialize APE identifier and comparator models."""
    # Build APE identifier
    ape_opt = [
        f'train.init_checkpoint={args.ape_checkpoint}',
        'model.model_vision.select_box_nums_for_evaluation=500',
        'model.model_vision.text_feature_bank_reset=True',
        'model.model_vision.backbone.net.xattn=True',
    ]
    identifier = build_ape(
        args.ape_config,
        args.confidence_thresh,
        ape_opt
    )
    
    # Build comparator model
    comparator, processor = get_model_and_processor(
        args.comparetor_config['model_type'],
        args.device
    )
    
    return identifier, comparator, processor


def process_image_pair(identifier, comparator, processor, img1_path, img2_path, output_dir, device):
    """Process a single image pair and save change mask."""
    # Load images
    img1 = imread(img1_path)
    img2 = imread(img2_path)

    # Get predictions from APE
    pred1 = extract_prediction_from_ape(identifier, img1_path, args.text_prompt)
    pred2 = extract_prediction_from_ape(identifier, img2_path, args.text_prompt)

    # Extract instance data
    def extract_instances(pred):
        return (
            pred['instances'].pred_boxes.tensor.cpu().numpy(),
            pred['instances'].pred_masks.cpu().numpy(),
            pred['instances'].scores.cpu().numpy()
        )
    
    img1_boxes, img1_masks, img1_scores = extract_instances(pred1)
    img2_boxes, img2_masks, img2_scores = extract_instances(pred2)

    # Combine masks
    combined_masks = np.concatenate([img1_masks, img2_masks], axis=0)
    img1_mask_num = len(img1_masks)

    # Instance-level change detection
    cmasks, img1_mask_num = instance_ceg(combined_masks, img1_mask_num, iou_threshold=1e-2)
    cmasks, img1_mask_num = bitemporal_match(
        img1, img2, cmasks, comparator, processor,
        img1_mask_num, change_confidence_threshold=135,
        device=device, model_config=args.comparetor_config
    )

    # Generate and save change mask
    change_mask = merge_masks(cmasks, img1.shape[:2])
    output_path = os.path.join(output_dir, os.path.basename(img1_path))
    imsave(output_path, change_mask)
    
    
def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize models
    identifier, comparator, processor = initialize_models(args)
    
    # Process each image pair
    process_image_pair(
            identifier, comparator, processor,
            args.input_image_1, args.input_image_2, args.output_dir, args.device
        )


if __name__ == "__main__":
    args = parse_arguments()
    main(args)