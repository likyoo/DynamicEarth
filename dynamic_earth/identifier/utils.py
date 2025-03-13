import torch
import numpy as np
from torchvision import transforms

from .segearth_ov_ext import SegEarth_OV


@torch.no_grad()
def get_identifier(
    model_type: str,
    device: str,
    name_list: list,
    processor_config: dict = None,
    model_config: dict = None,
) -> tuple:
    """
    Initializes the model and processor.

    Parameters:
    - model_type (str): Type of the model (e.g., 'SegEarth-OV').
    - device (str): Device to run the model on (e.g., 'cpu', 'cuda').
    - name_list (list): List of names to be processed.
    - processor_config (dict, optional): Configuration dictionary for the processor.
    - model_config (dict, optional): Configuration dictionary for the model.

    Returns:
    - tuple: A tuple containing the model and the processor.
    """

    if model_type == 'SegEarth-OV':
        # Default model configuration
        default_model_config = {
            'clip_type': 'CLIP',
            'vit_type': 'ViT-B/16',
            'model_type': 'SegEarth',
            'ignore_residual': True,
            'feature_up': True,
            'feature_up_cfg': {
                'model_name': 'jbu_one',
                'model_path': 'third_party/SegEarth_OV/simfeatup_dev/weights/xclip_jbu_one_million_aid.ckpt'
            },
            'cls_token_lambda': -0.3,
            'name_path': './class_name.txt'
        }
        
        # Update default processor config with provided values
        if model_config:
            default_model_config.update(model_config)
        
        # Default processor configuration
        default_processor_config = {
            'normalize_mean': [0.48145466, 0.4578275, 0.40821073],
            'normalize_std': [0.26862954, 0.26130258, 0.27577711],
            'resize': (448, 448)
        }
        
        # Update default processor config with provided values
        if processor_config:
            default_processor_config.update(processor_config)

        if model_type == 'SegEarth-OV':
            processor = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(default_processor_config['normalize_mean'], default_processor_config['normalize_std']),
                transforms.Resize(default_processor_config['resize'])
            ])
        
        # Save names to a file
        with open('./class_name.txt', 'w') as writer:
            writer.write('\n'.join(name_list))

        default_model_config.update({'device': device})
        model = SegEarth_OV(**default_model_config)
        return model, processor


@torch.no_grad()
def identify(
    img1: np.array,
    img2: np.array,
    mask_data: np.array,
    img1_mask_num: int,
    model: torch.nn.Module,
    processor: transforms.Compose,
    model_type: str = 'SegEarth-OV',
    device: str = 'cpu',
    is_instance_class: bool = True,
) -> np.array:
    """
    Identifies changes between two images using the model.

    Parameters:
    - img1 (np.array): T1 image.
    - img2 (np.array): T2 image.
    - mask_data (np.array): Mask data for predictions.
    - img1_mask_num (int): Number of masks for the first image.
    - model (torch.nn.Module): The model used for predictions.
    - processor (transforms.Compose): The processor used to prepare images.
    - model_type (str): Type of the model; defaults to 'SegEarth-OV'.
    - device (str): Device to run the model on; defaults to 'cpu'.
    - is_instance_class (bool): Whether to use instance class; defaults to True.

    Returns:
    - np.array: Array of masks indicating changes between img1 and img2.
    """
    
    if model_type == 'SegEarth-OV':
        img1_tensor = processor(img1).unsqueeze(0).to(device)
        img1_mask_classes = model.predict(img1_tensor, data_samples=None, proposal_masks=mask_data)
        
        img2_tensor = processor(img2).unsqueeze(0).to(device)
        img2_mask_classes = model.predict(img2_tensor, data_samples=None, proposal_masks=mask_data)
    else:
        raise NotImplementedError("Model type not supported.")
    
    # Only for binary classification
    if is_instance_class: # e.g. building
        change_instance_match = img1_mask_classes[:img1_mask_num] + img2_mask_classes[img1_mask_num:]
        change_idx = np.where((np.array(img1_mask_classes) != np.array(img2_mask_classes)) & 
                            np.array(change_instance_match).astype(bool))
    else:
        # e.g. tree, low-vegetation
        change_idx = np.where(np.array(img1_mask_classes) != np.array(img2_mask_classes))

    cmasks = np.array(mask_data)[change_idx]
    img1_mask_num = np.sum(change_idx[0] < img1_mask_num)
    
    return cmasks, img1_mask_num