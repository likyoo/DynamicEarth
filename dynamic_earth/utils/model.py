import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModel


@torch.no_grad()
def get_model_and_processor(
    model_type,
    # checkpoint,
    device,
    model_config=None,
    processor_config=None,
    ):
    
    if model_type == 'DINO':
        processor = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
        model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16').to(device)
    elif model_type == 'DINOv2':
        # processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base', do_resize=True, \
        #     size={"shortest_edge": 896}, do_center_crop=False)
        processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base', \
            do_resize=False, do_center_crop=False)
        model = AutoModel.from_pretrained('facebook/dinov2-base').to(device)
    
    return model, processor
