import math
import torch
import numpy as np
import torch.nn.functional as F
from skimage.filters.thresholding import threshold_otsu


def angle_to_cosine(angle):
    """Convert an angle in degrees to its cosine value."""
    assert 0 <= angle <= 180, "Angle must be between 0 and 180 degrees."
    return math.cos(math.radians(angle))


def cosine_to_angle(cosine):
    """Convert a cosine value to its corresponding angle in degrees."""
    assert -1 <= cosine <= 1, "Cosine value must be between -1 and 1."
    return math.degrees(math.acos(cosine))


def extract_features(image: np.ndarray, model, processor, model_config: dict, device: str) -> torch.Tensor:
    """
    Process the input image and extract features using the specified model.

    Args:
        image (np.ndarray): Input image as a NumPy array.
        model: Pre-trained model for feature extraction.
        processor: Pre-processing function for images.
        model_config (dict): Configuration dictionary containing model type, feature dimension, and patch size.
        device (str): Device to run the model on ('cpu' or 'cuda').

    Returns:
        torch.Tensor: Extracted feature tensor.
    """
    model_type = model_config['model_type']
    feature_dim = model_config['feature_dim']
    patch_size = model_config['patch_size']

    if model_type == 'DINO':
        processed_image = processor(image).unsqueeze(0).to(device)
        h, w = processed_image.shape[-2:]
        features = model.get_intermediate_layers(processed_image)[0]
        features = features[:, 1:].permute(0, 2, 1).view(1, feature_dim, h // patch_size, w // patch_size)
    elif model_type == 'DINOv2':
        processed_image = processor(images=image, return_tensors="pt").to(device)
        h, w = processed_image['pixel_values'].shape[-2:]
        features = model(**processed_image)
        features = features.last_hidden_state[:, 1:].permute(0, 2, 1).view(1, feature_dim, h // patch_size, w // patch_size)
    else:
        raise ValueError("Unsupported model type. Use 'DINO' or 'DINOv2'.")

    return features


@torch.no_grad()
def bitemporal_match(
    img1: np.ndarray,
    img2: np.ndarray,
    mask_data: np.ndarray,
    model,
    processor,
    img1_mask_num: int,
    model_config: dict = None,
    device='cpu',
    auto_threshold=False,
    change_confidence_threshold=145,
):
    """
    Perform a bitemporal change detection between two images using a specified model.
    This function is an extension of AnyChange. ``Segment any change. arXiv:2402.01188.``
    
    Args:
        img1 (np.ndarray): T1 image.
        img2 (np.ndarray): T2 image.
        mask_data (np.ndarray): Mask data for change detection (boolean array).
        model: Pre-trained model for feature extraction.
        processor: Pre-processing function for images.
        img1_mask_num (int): Number of masks from img1 to consider.
        model_config (dict, optional): Configuration dictionary for model parameters. 
            Defaults to {'model_type': 'DINO', 'feature_dim': 768, 'patch_size': 16}.
        device (str): Device to run the model on ('cpu' or 'cuda').
        auto_threshold (bool): Whether to compute threshold automatically.
        change_confidence_threshold (float): Threshold for change confidence.
    
    Returns:
        np.ndarray: Filtered mask data after change detection.
        int: Number of masks retained from img1 after filtering.
    """

    # Set default model configuration if none provided
    if model_config is None:
        model_config = {
            'model_type': 'DINO',
            'feature_dim': 768,
            'patch_size': 16
        }
    
    # Extract features for both images
    img1_embed = extract_features(img1, model, processor, model_config, device)
    img2_embed = extract_features(img2, model, processor, model_config, device)

    # Resize embeddings to match the original image dimensions
    H, W = img1.shape[:2]
    img1_embed = F.interpolate(img1_embed, size=(H, W), mode='bilinear', align_corners=True).squeeze_(0)
    img2_embed = F.interpolate(img2_embed, size=(H, W), mode='bilinear', align_corners=True).squeeze_(0)

    # Automatically compute the threshold if required
    if auto_threshold:
        cos_similarity = -F.cosine_similarity(img1_embed, img2_embed, dim=0)
        cos_similarity_flat = cos_similarity.reshape(-1).cpu().numpy()
        threshold = threshold_otsu(cos_similarity_flat)
        change_confidence_threshold = cosine_to_angle(threshold)

    def _latent_match(mask_data: np.ndarray, img1_embed: torch.Tensor, img2_embed: torch.Tensor):
        """Match latent features of images based on the provided mask."""
        change_confidence = torch.zeros(len(mask_data), dtype=torch.float32, device=device)

        for i, mask in enumerate(mask_data):
            binary_mask = torch.from_numpy(mask).to(device, dtype=torch.bool)
            t1_mask_embed = torch.mean(img1_embed[:, binary_mask], dim=-1)
            t2_mask_embed = torch.mean(img2_embed[:, binary_mask], dim=-1)
            score = -F.cosine_similarity(t1_mask_embed, t2_mask_embed, dim=0)
            change_confidence[i] += score

        # Keep masks where confidence exceeds the threshold
        keep_indices = change_confidence > angle_to_cosine(change_confidence_threshold)
        keep_indices = keep_indices.cpu().numpy()
        retained_mask_data = mask_data[keep_indices]
        retained_count_img1 = len(np.where(keep_indices[:img1_mask_num])[0])

        return retained_mask_data, retained_count_img1

    # Perform latent matching and return results
    filtered_mask_data, filtered_img1_mask_num = _latent_match(mask_data, img1_embed, img2_embed)
    return filtered_mask_data, filtered_img1_mask_num