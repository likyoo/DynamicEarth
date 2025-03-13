"""
Change Detection Evaluation Metric Calculation

This script implements a standardized evaluation pipeline for change detection tasks,
calculating common metrics including mIoU, OA, F1-score, etc.

python eval/evaluate.py --gt [GROUND_TRUTH_DIR] --pred [PREDICTION_DIR] --threshold [0.5]

"""

import os
from typing import Dict, Optional
import argparse

import cv2
import numpy as np
from tqdm import tqdm


class ChangeDetectionMetrics:
    """
    Change Detection Evaluation Metric Calculator

    Attributes:
        threshold (float): Binarization threshold (0-1 scale)
        eps (float): Numerical stability constant
        tp (float): Accumulated true positives
        tn (float): Accumulated true negatives
        fp (float): Accumulated false positives
        fn (float): Accumulated false negatives
        results (dict): Dictionary storing final evaluation metrics

    Methods:
        reset(): Resets all accumulators
        update(): Updates metric calculations with new batch
        compute(): Computes and returns all metrics
    """

    def __init__(self, threshold: float = 0.5) -> None:
        """
        Initialize metric calculator
        
        Args:
            threshold: Binarization threshold (0-1 scale), default 0.5
        """
        self.threshold = threshold * 255.0  # Convert to pixel value
        self.eps = 1e-7  # Numerical stability constant
        
        # Initialize accumulators
        self.tp = 0.0
        self.tn = 0.0
        self.fp = 0.0
        self.fn = 0.0
        
        # Store final results
        self.results: Optional[Dict[str, float]] = None

    def reset(self) -> None:
        """Resets all accumulators to zero"""
        self.tp = 0.0
        self.tn = 0.0
        self.fp = 0.0
        self.fn = 0.0

    def update(self, prediction: np.ndarray, target: np.ndarray) -> None:
        """
        Update metrics with new data pair
        
        Args:
            prediction: Model prediction (grayscale image, 0-255)
            target: Ground truth (grayscale image, 0-255)
        """
        # Convert to binary masks
        pred_binary = (prediction > self.threshold)
        target_binary = (target > self.threshold)

        # Update confusion matrix elements
        self.tp += np.sum(pred_binary & target_binary)
        self.tn += np.sum(~pred_binary & ~target_binary)
        self.fp += np.sum(pred_binary & ~target_binary)
        self.fn += np.sum(~pred_binary & target_binary)

    def compute(self) -> Dict[str, float]:
        """Compute and return all evaluation metrics"""
        # Calculate IoU for both classes
        iou_change = self.tp / (self.tp + self.fp + self.fn + self.eps)
        iou_nochange = self.tn / (self.tn + self.fp + self.fn + self.eps)
        
        # Calculate mean IoU
        miou = 0.5 * (iou_change + iou_nochange)
        
        # Calculate overall accuracy
        oa = (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn + self.eps)
        
        # Calculate precision/recall/F1-score
        precision = self.tp / (self.tp + self.fp + self.eps)
        recall = self.tp / (self.tp + self.fn + self.eps)
        f1_score = (2 * precision * recall) / (precision + recall + self.eps)

        # Organize results
        self.results = {
            'miou': miou,
            'oa': oa,
            'iou_change': iou_change,
            'iou_nochange': iou_nochange,
            'f1_score_change': f1_score,
            'precision_change': precision,
            'recall_change': recall
        }
        return self.results


def evaluate_metrics(
    ground_truth_dir: str,
    prediction_dir: str,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Execute full evaluation pipeline
    
    Args:
        ground_truth_dir: Path to ground truth directory
        prediction_dir: Path to prediction directory
        threshold: Binarization threshold (0-1 scale)
    
    Returns:
        Dictionary containing evaluation metrics
    """
    # Initialize metric calculator
    metric_calculator = ChangeDetectionMetrics(threshold=threshold)
    
    # Get sorted file list
    image_list = sorted(os.listdir(prediction_dir))
    
    # Process all image pairs
    for filename in tqdm(image_list, desc="Processing Images"):
        # Construct file paths
        pred_path = os.path.join(prediction_dir, filename)
        gt_path = os.path.join(ground_truth_dir, filename)
        
        try:
            # Read images as grayscale
            pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
            gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            
            # Validate inputs
            if pred is None or gt is None:
                raise ValueError(f"Invalid image file: {filename}")
            if pred.shape != gt.shape:
                raise ValueError(f"Size mismatch: {filename}")
            
            # Update metrics
            metric_calculator.update(pred, gt)
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            continue
    
    # Return computed metrics
    return metric_calculator.compute()


if __name__ == "__main__":
    # Configure argument parser
    parser = argparse.ArgumentParser(description="Change Detection Evaluation")
    parser.add_argument('--gt', type=str, required=True, help="Path to ground truth directory")
    parser.add_argument('--pred', type=str, required=True, help="Path to prediction directory")
    parser.add_argument('--threshold', type=float, default=0.5, help="Binarization threshold (0-1)")
    args = parser.parse_args()

    # Run evaluation
    results = evaluate_metrics(
        ground_truth_dir=args.gt,
        prediction_dir=args.pred,
        threshold=args.threshold
    )

    # Print formatted results
    print("\nEvaluation Results:")
    for metric, value in results.items():
        print(f"{metric:15}: {value:.4f}")
