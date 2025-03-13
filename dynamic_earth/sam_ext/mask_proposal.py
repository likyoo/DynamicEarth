import torch
import numpy as np
from torchvision.ops.boxes import batched_nms
from skimage.exposure import match_histograms
from segment_anything.utils.amg import MaskData

from .simple_maskgen import SimpleMaskGenerator


class MaskProposal:
    def __init__(self):
        self.set_hyperparameters()

    def set_hyperparameters(self, **kwargs):
        self.match_hist = kwargs.get('match_hist', False)
        self.area_thresh = kwargs.get('area_thresh', 0.8)

    def make_mask_generator(self, **kwargs):
        self.maskgen = SimpleMaskGenerator(**kwargs)
    
    def proposal(self, img):
        h, w = img.shape[:2]
        mask_data = self.maskgen.simple_generate(img)
        mask_data.filter((mask_data['areas'] / (h * w)) < self.area_thresh) # TODO: merge into postprocess_small_regions
        return mask_data, len(mask_data['rles'])

    def forward(self, img1, img2):

        if self.match_hist:
            img2 = match_histograms(image=img2, reference=img1, channel_axis=-1).astype(np.uint8)

        data = MaskData()
        mask_num = []
        for im in [img1, img2]:
            d, l = self.proposal(im)
            data.cat(d)
            mask_num.append(l)

        keep = batched_nms(
            data["boxes"].float(),
            data["iou_preds"],
            torch.zeros_like(data["boxes"][:, 0]),
            iou_threshold=self.maskgen.box_nms_thresh,
        )
        keep = keep.sort()[0] # Note: The sorting here is for the subsequent separate processing of img1_mask and img2_mask
        data.filter(keep)

        # return mask data and mask number of img1
        return data, len(keep[keep < mask_num[0]])