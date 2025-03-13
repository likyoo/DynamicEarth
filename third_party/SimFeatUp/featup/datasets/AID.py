import random
import os
from os.path import join

import numpy as np
import torch
import torch.multiprocessing
from PIL import Image
from torch.utils.data import Dataset

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


class Million_AID(Dataset):
    def __init__(self,
                 root,
                 transform):
        super().__init__()
        self.root = join(root, "test")
        self.transform = transform

        self.image_files = []
        for file_name in os.listdir(self.root):
            self.image_files.append(join(self.root, file_name))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image_path = self.image_files[index]
        seed = np.random.randint(42)
        batch = {}

        random.seed(seed)
        torch.manual_seed(seed)
        img = self.transform(Image.open(image_path).convert("RGB"))
        batch["img"] = img
        batch["img_path"] = image_path

        return batch