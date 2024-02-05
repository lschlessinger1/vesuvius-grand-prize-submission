import numpy as np
import torch
import torch.nn.functional as F
import zarr
from skimage.util import img_as_ubyte
from torch.utils.data import Dataset

from vesuvius_challenge_rnd.scroll_ink_detection.ink_detection.data.patch_memmap_dataset import (
    _verify_patch_positions,
    cut_and_paste_depth,
)

ArrayLike = np.ndarray | np.memmap | zarr.Array


class BlockZConstantDataset(Dataset):
    """A block dataset for a single segment."""

    def __init__(
        self,
        img_stack: ArrayLike,
        patch_positions: np.ndarray,
        size: int,
        z_start: int,
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
        pad_y, pad_x = padding_yx
        img_stack_z, img_stack_y, img_stack_x = img_stack.shape
        _verify_patch_positions((img_stack_y + pad_y, img_stack_x + pad_x), patch_positions)
        if non_ink_ignored_patches is not None:
            if len(non_ink_ignored_patches) != len(patch_positions):
                raise ValueError(
                    f"`non_ink_ignored_patches` length ({len(non_ink_ignored_patches)}) must be the same length as the `patch_positions` ({len(patch_positions)})."
                )

        if not (0 <= z_start <= img_stack_z):
            raise ValueError(
                f"z-start be between 0 (inclusive) and image depth {img_stack_z} (inclusive)."
            )

        if not (0 < z_extent <= img_stack_z):
            raise ValueError(
                f"z-start be between 0 (exclusive) and image depth {img_stack_z} (inclusive)."
            )

        if z_start + z_extent > img_stack_z:
            raise ValueError(
                f"z_start ({z_start}) + z_extent ({z_extent}) cannot exceed image depth ({img_stack_z})."
            )

        self.img_stack = img_stack
        self.patch_positions = patch_positions
        self.z_start = z_start
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

    @property
    def z_end(self) -> int:
        return self.z_start + self.z_extent

    def __len__(self) -> int:
        return len(self.patch_positions)

    def _preprocess_img_patch(self, img_patch: np.ndarray) -> np.ndarray:
        # Pad image.
        pad_y = self.size - img_patch.shape[1]
        pad_x = self.size - img_patch.shape[2]
        img_patch = np.pad(img_patch, [(0, 0), (0, pad_y), (0, pad_x)], constant_values=0)

        # Rescale intensities and convert to uint8.
        img_patch = img_as_ubyte(img_patch)

        # Clip intensities
        img_patch = np.clip(img_patch, self.clip_min, self.clip_max)

        # Swap axes (move depth dimension to be last).
        img_patch = np.moveaxis(img_patch, 0, -1)

        return img_patch

    def __getitem__(self, idx: int):
        x1, y1, x2, y2 = self.patch_positions[idx]
        if self.z_reverse:  # Reverse along z-dimension if necessary
            z1 = -self.z_end
            z2 = -self.z_start
            z_step = -1
        else:
            z1 = self.z_start
            z2 = self.z_end
            z_step = 1

        img_patch = self.img_stack[z1:z2, y1:y2, x1:x2][::z_step]
        img_patch = self._preprocess_img_patch(img_patch)
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

    def _transform_if_needed(
        self, image: np.ndarray, label: np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.transform:
            data = self.transform(image=image, mask=label)
            image = data["image"].unsqueeze(0)
            label = data["mask"]
            label = F.interpolate(
                label.unsqueeze(0),
                (self.size // self.label_downscale, self.size // self.label_downscale),
            ).squeeze(0)
        return image, label
