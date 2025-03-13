"""
Change Detection (SECOND dataset) Evaluation Metric Calculation

This script implements a standardized evaluation pipeline for change detection tasks,
calculating common metrics including mIoU, OA, F1-score, etc.

python eval/evaluate_second.py --pred [PREDICTION_DIR] --gt1 [GROUND_TRUTH_DIR1] --gt2 [GROUND_TRUTH_DIR2] --class-name [CLASS_NAME] --threshold [0.5]

"""

import os
from typing import Dict, Optional
import argparse

import cv2
import numpy as np
from tqdm import tqdm

from evaluate import ChangeDetectionMetrics


def main():
    """Main execution flow"""
    # Configuration settings
    CLASS_MAPPING = {
        'background': 0,
        'water': 1,
        'ground': 2,
        'low vegetation': 3,
        'tree': 4,
        'building': 5,
        'playground': 6,
    }

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Semantic Change Detection Evaluation')
    parser.add_argument('--pred', type=str, required=True, help='Prediction directory path')
    parser.add_argument('--gt1', type=str, required=True, help='First time point labels directory')
    parser.add_argument('--gt2', type=str, required=True, help='Second time point labels directory')
    parser.add_argument('--class-name', type=str, default='playground', 
                       choices=list(CLASS_MAPPING.keys()), help='Target class for change detection')
    parser.add_argument('--threshold', type=float, default=0.5, 
                       help='Binarization threshold (0-1 scale)')
    args = parser.parse_args()

    # Initialize metric calculator
    metric = ChangeDetectionMetrics(threshold=args.threshold)
    class_id = CLASS_MAPPING[args.class_name]

    # Process all images
    image_list = sorted(os.listdir(args.pred))
    for filename in tqdm(image_list, desc='Evaluating Predictions'):
        try:
            # Load data
            pred_path = os.path.join(args.pred, filename)
            label1_path = os.path.join(args.gt1, filename)
            label2_path = os.path.join(args.gt2, filename)

            pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
            label1 = cv2.imread(label1_path, cv2.IMREAD_ANYCOLOR)
            label2 = cv2.imread(label2_path, cv2.IMREAD_ANYCOLOR)

            # Validate inputs
            if pred.shape != label1.shape or pred.shape != label2.shape:
                raise ValueError(f"Size mismatch in file: {filename}")

            # Generate change label
            change_label = ((label1 == class_id) | (label2 == class_id)).astype(np.uint8) * 255
            
            # Update metrics
            metric.update(pred, change_label)

        except Exception as e:
            print(f"Skipped {filename}: {str(e)}")
            continue

    # Output results
    results = metric.compute()
    print("\nEvaluation Results:")
    for metric_name, value in results.items():
        print(f"{metric_name:15}: {value:.4f}")


if __name__ == "__main__":
    main()
