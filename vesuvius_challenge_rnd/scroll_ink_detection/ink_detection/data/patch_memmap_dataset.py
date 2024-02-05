import random

import numpy as np
import torch.nn.functional as F
from skimage.util import img_as_ubyte
from torch.utils.data import Dataset


class PatchMemMapDataset(Dataset):
    """A patch memory-mapped dataset for a single segment."""

    def __init__(
        self,
        img_stack_ref: np.memmap,
        patch_positions: np.ndarray,
        size: int,
        z_extent: int = 30,
        label_downscale: int = 4,
        labels=None,
        transform=None,
        augment: bool = False,
        non_ink_ignored_patches: np.ndarray | None = None,
        ignore_idx: int = -100,
        min_crop_num_offset: int = 8,
        z_reverse: bool = False,
        padding_yx: tuple[int, int] = (0, 0),
        clip_min: int = 0,
        clip_max: int = 255,
    ):
        # pad_y, pad_x = padding_yx
        # img_stack_y, img_stack_x = img_stack_ref.shape[1:3]
        # _verify_patch_positions((img_stack_y + pad_y, img_stack_x + pad_x), patch_positions)
        _verify_patch_positions(img_stack_ref.shape[:2], patch_positions)
        if non_ink_ignored_patches is not None:
            if len(non_ink_ignored_patches) != len(patch_positions):
                raise ValueError(
                    f"`non_ink_ignored_patches` length ({len(non_ink_ignored_patches)}) must be the same length as the `patch_positions` ({len(patch_positions)})."
                )
        self.img_stack_ref = img_stack_ref
        self.patch_positions = patch_positions
        self.z_extent = z_extent
        self.size = size
        self.labels = labels
        self.label_downscale = label_downscale
        self.non_ink_ignored_patches = non_ink_ignored_patches
        self.ignore_idx = ignore_idx
        self.z_reverse = z_reverse

        self.transform = transform
        self.augment = augment
        self.min_crop_num_offset = min_crop_num_offset
        self.clip_min = clip_min
        self.clip_max = clip_max

    def __len__(self):
        return len(self.patch_positions)

    def _preprocess_img_patch(self, img_patch: np.ndarray) -> np.ndarray:
        # Pad image.
        pad_y = self.size - img_patch.shape[1]
        pad_x = self.size - img_patch.shape[2]
        img_patch = np.pad(img_patch, [(0, 0), (0, pad_y), (0, pad_x)], constant_values=0)

        # Rescale intensities and convert to uint8.
        img_patch = img_as_ubyte(img_patch)

        # Reverse along z-dimension if necessary
        if self.z_reverse:
            img_patch = img_patch[::-1]

        # Clip intensities
        img_patch = np.clip(img_patch, self.clip_min, self.clip_max)

        # Swap axes (move depth dimension to be last).
        img_patch = np.moveaxis(img_patch, 0, -1)

        return img_patch

    def __getitem__(self, idx: int):
        x1, y1, x2, y2 = self.patch_positions[idx]
        img_patch = self.img_stack_ref[y1:y2, x1:x2]
        label_patch = self.labels[y1:y2, x1:x2, None]
        if self.non_ink_ignored_patches is not None:
            non_ink_is_ignored = self.non_ink_ignored_patches[idx]
            if non_ink_is_ignored:
                # Replace all 0s (non-ink) with ignore_idx
                label_patch = np.where(label_patch == 0, self.ignore_idx, label_patch)

        if not self.augment:
            img_patch, label_patch = self._transform_if_needed(img_patch, label_patch)
            return img_patch, label_patch, self.patch_positions[idx]
        else:
            img_patch = cut_and_paste_depth(
                img_patch, self.z_extent, min_crop_num_offset=self.min_crop_num_offset
            )
            img_patch, label_patch = self._transform_if_needed(img_patch, label_patch)
            return img_patch, label_patch

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


class PatchMemMapDatasetUnlabeled(PatchMemMapDataset):
    """A patch memory-mapped dataset for a single segment."""

    def __init__(
        self,
        img_stack_ref: np.memmap,
        patch_positions: np.ndarray,
        size: int,
        z_extent: int = 30,
        transform=None,
        augment: bool = False,
    ):
        super().__init__(
            img_stack_ref,
            patch_positions,
            size,
            z_extent=z_extent,
            transform=transform,
            augment=augment,
        )

    def __getitem__(self, idx: int):
        x1, y1, x2, y2 = self.patch_positions[idx]
        img_patch = self.img_stack_ref[y1:y2, x1:x2]

        if not self.augment:
            img_patch = self._transform_image_if_needed(img_patch)
            return img_patch, self.patch_positions[idx]
        else:
            img_patch = cut_and_paste_depth(
                img_patch, self.z_extent, min_crop_num_offset=self.min_crop_num_offset
            )
            img_patch = self._transform_image_if_needed(img_patch)
            return img_patch

    def _transform_image_if_needed(self, image):
        if self.transform:
            data = self.transform(image=image)
            image = data["image"].unsqueeze(0)
        return image


def cut_and_paste_depth(
    image: np.ndarray, z_extent: int, p: float = 0.4, min_crop_num_offset: int = 8
) -> np.ndarray:
    image_tmp = np.zeros_like(image)

    # Random crop.
    cropping_num = random.randint(z_extent - min_crop_num_offset, z_extent)

    start_idx = random.randint(0, z_extent - cropping_num)
    crop_indices = np.arange(start_idx, start_idx + cropping_num)

    start_paste_idx = random.randint(0, z_extent - cropping_num)

    tmp = np.arange(start_paste_idx, cropping_num)
    np.random.shuffle(tmp)

    cutout_idx = random.randint(0, 2)
    temporal_random_cutout_idx = tmp[:cutout_idx]

    # Random paste.
    image_tmp[..., start_paste_idx : start_paste_idx + cropping_num] = image[..., crop_indices]

    # Random cutout.
    if random.random() > p:
        image_tmp[..., temporal_random_cutout_idx] = 0

    image = image_tmp
    return image


def _verify_patch_positions(img_xy_shape: tuple[int, int], patch_positions: np.ndarray) -> None:
    if patch_positions.ndim != 2 or patch_positions.shape[1] != 4:
        raise ValueError(
            f"Expected patch positions to be N x 4. Found shape {patch_positions.shape}"
        )

    # Check x1 < x2 and y1 < y2
    x1 = patch_positions[:, 0]
    y1 = patch_positions[:, 1]
    x2 = patch_positions[:, 2]
    y2 = patch_positions[:, 3]
    if not (x1 < x2).all():
        raise ValueError("x1 must be less than x2.")

    if not (y1 < y2).all():
        raise ValueError("y1 must be less than y2.")

    if x1.min() < 0:
        raise ValueError("x1 must be nonnegative.")

    if y1.min() < 0:
        raise ValueError("y1 must be nonnegative.")

    # Check x2 <= img width and y2 <= img height
    img_width = img_xy_shape[1]
    if x2.max() > img_width:
        raise ValueError(f"x2 must be less than image width {img_width}. Found max x2 {x2.max()}.")

    img_height = img_xy_shape[0]
    if y2.max() > img_height:
        raise ValueError(
            f"y2 must be less than image height {img_height}. Found y2 max {y2.max()}."
        )
