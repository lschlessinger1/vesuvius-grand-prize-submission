import logging
from pathlib import Path

import numpy as np
from patchify import patchify
from pytorch_lightning.utilities import CombinedLoader
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
from torch.utils.data import ConcatDataset, DataLoader
from tqdm.contrib import tzip

from vesuvius_challenge_rnd import REPO_DIR, SCROLL_DATA_DIR
from vesuvius_challenge_rnd.patching import patch_index_to_pixel_position
from vesuvius_challenge_rnd.scroll_ink_detection.ink_detection.data.patch_memmap_dataset import (
    PatchMemMapDatasetUnlabeled,
)
from vesuvius_challenge_rnd.scroll_ink_detection.ink_detection.data.scroll_patch_data_module import (
    ScrollPatchDataModule,
    ScrollSegmentsDataType,
    create_dataset,
    create_non_ink_mask,
    create_train_patch_positions,
    create_val_patch_positions,
    generate_dataset_inputs,
)


class SemiSupervisedScrollPatchDataModule(ScrollPatchDataModule):
    def __init__(
        self,
        train_segment_ids: ScrollSegmentsDataType,
        val_segment_id: str | tuple[str, str],
        data_dir: Path = SCROLL_DATA_DIR,
        ink_label_dir: Path = REPO_DIR / "external" / "youssef-nader-first-letters" / "labels",
        z_min: int = 15,
        z_max: int = 45,
        size: int = 64,
        tile_size: int = 256,
        patch_stride_train: int = 32,
        patch_stride_val: int = 32,
        downsampling: int | None = None,
        batch_size: int = 256,
        num_workers: int = 0,
        blur_ink_labels: bool = False,
        ink_labels_blur_kernel_size: int = 17,
        ink_dilation_kernel_size: int = 256,
    ):
        super().__init__(
            train_segment_ids,
            val_segment_id,
            data_dir=data_dir,
            ink_label_dir=ink_label_dir,
            z_min=z_min,
            z_max=z_max,
            size=size,
            tile_size=tile_size,
            patch_stride_train=patch_stride_train,
            patch_stride_val=patch_stride_val,
            downsampling=downsampling,
            batch_size=batch_size,
            num_workers=num_workers,
            blur_ink_labels=blur_ink_labels,
            ink_labels_blur_kernel_size=ink_labels_blur_kernel_size,
            ink_dilation_kernel_size=ink_dilation_kernel_size,
        )

    def setup(self, stage: str) -> None:
        """Set up the data for the given stage.

        Args:
            stage (str): The stage for which to set up the data (e.g., "fit").
        """
        if stage == "fit" or stage is None:
            img_stack_refs_train, segment_masks_train, ink_masks_train, _ = generate_dataset_inputs(
                self.segments_train,
                self.z_min,
                self.z_max,
                self.tile_size,
                self.ink_label_dir,
                should_blur_ink_labels=self.blur_ink_labels,
                ink_labels_blur_kernel_size=self.ink_label_blur_kernel_size,
            )

            train_patch_positions_labeled = create_train_patch_positions(
                img_stack_refs_train,
                segment_masks_train,
                ink_masks_train,
                self.ink_dilation_kernel_size,
                self.size,
                self.patch_stride_train,
            )

            train_patch_positions_unlabeled = create_train_patch_positions_unlabeled(
                img_stack_refs_train,
                segment_masks_train,
                ink_masks_train,
                self.ink_dilation_kernel_size,
                self.size,
                self.patch_stride_train,
            )

            img_stack_refs_val, segment_masks_val, ink_masks_val, _ = generate_dataset_inputs(
                [self.segment_val], self.z_min, self.z_max, self.tile_size, self.ink_label_dir
            )
            val_patch_positions = create_val_patch_positions(
                img_stack_refs_val,
                segment_masks_val,
                ink_masks_val,
                self.ink_dilation_kernel_size,
                self.size,
                self.patch_stride_val,
            )

            self.data_train_labeled = create_dataset(
                img_stack_refs_train,
                train_patch_positions_labeled,
                ink_masks_train,
                self.size,
                self.z_extent,
                augment=True,
                transform=self.train_transform(),
            )

            self.data_train_unlabeled = create_unlabeled_dataset(
                img_stack_refs_train,
                train_patch_positions_unlabeled,
                self.size,
                self.z_extent,
                augment=True,
                transform=self.train_transform(),
            )

            self.data_val = create_dataset(
                img_stack_refs_val,
                val_patch_positions,
                ink_masks_val,
                self.size,
                self.z_extent,
                augment=False,
                transform=self.val_transform(),
            )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        labeled_data_loader = DataLoader(
            self.data_train_labeled,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
        )
        unlabeled_data_loader = DataLoader(
            self.data_train_unlabeled,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
        )
        iterables = {
            "labeled": labeled_data_loader,
            "unlabeled": unlabeled_data_loader,
        }
        return CombinedLoader(iterables, mode="min_size")


def create_unlabeled_dataset(
    img_stack_refs: list[np.memmap],
    patch_position_list: list,
    size: int,
    z_extent: int,
    augment: bool = False,
    transform=None,
) -> ConcatDataset:
    datasets = []
    for img_stack_ref, patch_positions in zip(img_stack_refs, patch_position_list):
        datasets.append(
            PatchMemMapDatasetUnlabeled(
                img_stack_ref,
                np.array(patch_positions),
                size,
                z_extent=z_extent,
                augment=augment,
                transform=transform,
            )
        )
    return ConcatDataset(datasets)


def create_train_patch_positions_unlabeled(
    img_stack_refs: list[np.memmap],
    segment_masks: list[np.ndarray],
    ink_masks: list[np.ndarray],
    dilation_kernel_size: int,
    size: int,
    stride: int,
):
    train_patch_positions = []
    pbar = tzip(img_stack_refs, segment_masks, ink_masks, desc="Creating train patches...")
    for img_stack_ref, segment_mask, ink_mask in pbar:
        # Pad ink mask
        pad_y = segment_mask.shape[0] - ink_mask.shape[0]
        pad_x = segment_mask.shape[1] - ink_mask.shape[1]
        ink_mask = np.pad(ink_mask, [(0, pad_y), (0, pad_x)], constant_values=0)

        # Create non-ink mask
        non_ink_mask = create_non_ink_mask(ink_mask, dilation_kernel_size=dilation_kernel_size)

        # Create patch positions
        patch_shape = (size, size)
        usable_patch_map = create_usable_patch_map_unlabeled_data(
            segment_mask, ink_mask, non_ink_mask, patch_shape=patch_shape, patch_stride=stride
        )
        usable_patch_position_arr = create_usable_patch_position_arr(
            usable_patch_map, patch_shape=patch_shape, patch_stride=stride
        ).tolist()

        logging.info(f"Created {len(usable_patch_position_arr)} training patch positions")
        train_patch_positions.append(usable_patch_position_arr)

    return train_patch_positions


def create_usable_patch_map_unlabeled_data(
    segment_mask: np.ndarray,
    ink_mask: np.ndarray,
    non_ink_mask: np.ndarray,
    patch_shape: tuple[int, int],
    patch_stride: int,
    ink_thresh: float = 0.05,
) -> np.ndarray:
    if (segment_mask.shape != ink_mask.shape) or (segment_mask.shape != non_ink_mask.shape):
        raise ValueError(
            f"Shapes of segment mask, ink mask, and non-ink mask must match. Found shapes "
            f"{segment_mask.shape}, {ink_mask.shape}, {non_ink_mask.shape}, and respectively."
        )
    segment_patches = patchify(segment_mask, patch_size=patch_shape, step=patch_stride)
    ink_patches = patchify(ink_mask, patch_size=patch_shape, step=patch_stride)
    non_ink_patches = patchify(non_ink_mask, patch_size=patch_shape, step=patch_stride)
    # Create a usable patch map where true indicates that the patch is usable for training/validation/test and
    # false indicates there is not enough papyrus.
    usable_patch_map = np.empty(shape=(segment_patches.shape[:2]), dtype=bool)
    for i in range(segment_patches.shape[0]):
        for j in range(segment_patches.shape[1]):
            segment_patch = segment_patches[i, j]
            ink_patch = ink_patches[i, j]
            non_ink_patch = non_ink_patches[i, j]

            no_overlap_with_non_segment_region = not np.any(segment_patch == 0)
            ink_patch_has_no_ink = not np.any(ink_patch > ink_thresh)
            non_ink_patch_has_no_non_ink = not np.any(non_ink_patch == 1)

            usable_patch_map[i, j] = (
                no_overlap_with_non_segment_region
                and ink_patch_has_no_ink
                and non_ink_patch_has_no_non_ink
            )
    return usable_patch_map


def create_usable_patch_position_arr(
    usable_patch_map: np.ndarray, patch_shape: tuple[int, int], patch_stride: int
) -> np.ndarray:
    usable_patch_positions = []
    for i in range(usable_patch_map.shape[0]):
        for j in range(usable_patch_map.shape[1]):
            if usable_patch_map[i, j]:
                (y0, x0), (y1, x1) = patch_index_to_pixel_position(i, j, patch_shape, patch_stride)
                usable_patch_positions.append((x0, y0, x1, y1))

    return np.array(usable_patch_positions)
