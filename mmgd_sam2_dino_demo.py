import os
import argparse
import torch
import numpy as np
from skimage.io import imread, imsave
from torchvision.ops.boxes import batched_nms

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from dynamic_earth.utils import get_model_and_processor
from dynamic_earth.comparator.ins_ceg import instance_ceg
from dynamic_earth.comparator.bi_match import bitemporal_match
from dynamic_earth.identifier.mmgd_ext import build_mmdet_model, extract_bbox_and_cls_from_mmgd


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

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="DynamicEarth Demo")
    parser.add_argument("--input_image_1", type=str, required=True,
                        help="Path to directory containing first set of images")
    parser.add_argument("--input_image_2", type=str, required=True,
                        help="Path to directory containing second set of images")
    parser.add_argument("--output_dir", "-o", type=str, default="outputs",
                        help="Output directory for change masks")
    parser.add_argument("--text_prompt", type=str, 
                        default="roof. rooftop. building. house. apartment. residential. factory",
                        help="Text prompts for identification")
    parser.add_argument("--mmgd_config", type=str,
                        default="dynamic_earth/identifier/mmgd_ext/grounding_dino_swin-l_pretrain_all.py",
                        help="Path to MM-GroundingDINO config file")
    parser.add_argument("--mmgd_checkpoint", type=str,
                        default="weights/grounding_dino_swin-l_pretrain_all-56d69e78.pth",
                        help="Path to MM-GroundingDINO checkpoint")
    parser.add_argument("--score_thresh", type=float, default=0.1,
                        help="Score threshold for predictions")
    parser.add_argument("--nms_iou_thresh", type=float, default=0.7,
                        help="NMS IoU threshold")
    parser.add_argument("--sam2_config", type=str,
                        default="configs/sam2.1/sam2.1_hiera_l.yaml",
                        help="Path to SAM2 config file")
    parser.add_argument("--sam2_checkpoint", type=str,
                        default="weights/sam2.1_hiera_large.pt",
                        help="Path to SAM2 checkpoint")
    parser.add_argument("--comparetor_config", type=dict,
                        default={'model_type': 'DINO',
                                 'feature_dim': 768,
                                 'patch_size': 16})
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run on")
    return parser.parse_args()

def initialize_models(args):
    """Initialize models and predictors."""
    # Build MM-GroundingDINO identifier
    identifier = build_mmdet_model(
        model=args.mmgd_config,
        weights=args.mmgd_checkpoint, 
        device=args.device
    )
    
    # Build SAM2 predictor
    sam2_model = build_sam2(args.sam2_config, args.sam2_checkpoint, args.device)
    sam2_predictor = SAM2ImagePredictor(sam2_model)
    
    # Build comparator model
    comparator_model, comparator_processor = get_model_and_processor(
        args.comparetor_config['model_type'],
        args.device
    )
    
    return identifier, sam2_predictor, comparator_model, comparator_processor

def process_image_pair(args, identifier, sam2_predictor, comparator_model, comparator_processor, img_pair_paths):
    """Process a single image pair and generate change mask."""
    img1_path, img2_path = img_pair_paths
    img1 = imread(img1_path)
    img2 = imread(img2_path)

    # Get predictions from identifier
    predictions = extract_bbox_and_cls_from_mmgd(
        identifier, [img1_path, img2_path],
        texts=args.text_prompt, 
        custom_entities=True
    )
    pred1, pred2 = predictions['predictions']

    # Filter predictions
    img1_boxes, img1_scores = filter_predictions(pred1, args.score_thresh, args.nms_iou_thresh, args.device)
    img2_boxes, img2_scores = filter_predictions(pred2, args.score_thresh, args.nms_iou_thresh, args.device)

    # Handle empty predictions
    if len(img1_boxes) == 0 and len(img2_boxes) == 0:
        return np.zeros(img1.shape[:2], dtype=np.uint8)
    
    if len(img1_boxes) == 0 or len(img2_boxes) == 0:
        img_boxes = img1_boxes if img1_boxes.size else img2_boxes
        img = img1 if img1_boxes.size else img2
        
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            sam2_predictor.set_image(img)
            masks, _, _ = sam2_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=img_boxes,
                mask_input=None,
                multimask_output=False
            )
        return merge_masks(masks.squeeze(1).astype(np.uint8), img.shape[:2])

    # Process both images
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        # Process first image
        sam2_predictor.set_image(img1)
        img1_masks, _, _ = sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=img1_boxes,
            mask_input=None,
            multimask_output=False
        )

        # Process second image
        sam2_predictor.set_image(img2)
        img2_masks, _, _ = sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=img2_boxes,
            mask_input=None,
            multimask_output=False
        )

    # Post-process masks
    # convert the shape to (n, H, W)
    if img1_masks.ndim == 4:
        img1_masks = img1_masks.squeeze(1)
    if img2_masks.ndim == 4:
        img2_masks = img2_masks.squeeze(1)
    combined_masks = np.concatenate([img1_masks, img2_masks], axis=0)
    
    # Instance-level change detection
    cmasks, img1_mask_num = instance_ceg(combined_masks, len(img1_masks), iou_threshold=1e-2)
    cmasks, _ = bitemporal_match(
        img1, img2, cmasks, 
        comparator_model,
        comparator_processor,
        img1_mask_num, 
        change_confidence_threshold=130,
        device=args.device,
        model_config=args.comparetor_config
    )
    
    return merge_masks(cmasks, img1.shape[:2])


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize models
    identifier, sam2_predictor, comparator_model, comparator_processor = initialize_models(args)
    
    # Process each image pair
    change_mask = process_image_pair(
        args, identifier, sam2_predictor, comparator_model, comparator_processor, 
        (args.input_image_1, args.input_image_2))
    
    # save the final change mask
    imsave(os.path.join(args.output_dir, os.path.basename(args.input_image_1)), change_mask)
 

if __name__ == "__main__":
    args = parse_arguments()
    main(args)