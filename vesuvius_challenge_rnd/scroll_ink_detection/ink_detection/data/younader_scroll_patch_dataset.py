import random

import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(
        self,
        images,
        size: int,
        z_extent: int = 30,
        label_downscale: int = 4,
        xyxys=None,
        labels=None,
        transform=None,
    ):
        self.images = images
        self.z_extent = z_extent
        self.size = size
        self.labels = labels
        self.label_downscale = label_downscale

        self.transform = transform
        self.xyxys = xyxys

    def __len__(self):
        return len(self.images)

    def fourth_augment(self, image: np.ndarray) -> np.ndarray:
        image_tmp = np.zeros_like(image)
        cropping_num = random.randint(self.z_extent - 8, self.z_extent)

        start_idx = random.randint(0, self.z_extent - cropping_num)
        crop_indices = np.arange(start_idx, start_idx + cropping_num)

        start_paste_idx = random.randint(0, self.z_extent - cropping_num)

        tmp = np.arange(start_paste_idx, cropping_num)
        np.random.shuffle(tmp)

        cutout_idx = random.randint(0, 2)
        temporal_random_cutout_idx = tmp[:cutout_idx]

        image_tmp[..., start_paste_idx : start_paste_idx + cropping_num] = image[..., crop_indices]

        if random.random() > 0.4:
            image_tmp[..., temporal_random_cutout_idx] = 0
        image = image_tmp
        return image

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.xyxys is not None:
            xy = self.xyxys[idx]
            image, label = self._transform_if_needed(image, label)
            return image, label, xy
        else:
            image = self.fourth_augment(image)
            image, label = self._transform_if_needed(image, label)
            return image, label

    def _transform_if_needed(self, image, label):
        if self.transform:
            data = self.transform(image=image, mask=label)
            image = data["image"].unsqueeze(0)
            label = data["mask"]
            label = F.interpolate(
                label.unsqueeze(0),
                (self.size // self.label_downscale, self.size // self.label_downscale),
            ).squeeze(0)
        return image, label
