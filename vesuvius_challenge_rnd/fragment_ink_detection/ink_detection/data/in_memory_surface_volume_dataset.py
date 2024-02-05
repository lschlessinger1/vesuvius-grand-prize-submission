import bisect
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path

import numpy as np
from patchify import patchify
from torch.utils.data import Dataset
from tqdm import tqdm

from vesuvius_challenge_rnd.data.volumetric_segment import VolumetricSegment
from vesuvius_challenge_rnd.patching import patch_index_to_pixel_position


class SurfaceVolumeDataset(Dataset, ABC):
    """A dataset class to handle surface volume data.

    This class is responsible for loading and managing volumetric segments
    for training, validation, and testing purposes. It supports operations
    like patch creation and indexing based on usability criteria.

    Attributes:
        data_dir (Path): Directory containing the data.
        segment_names (list[str]): List of segment names to load.
        z_start (int): Minimum z-slice to include.
        z_end (int): Maximum z-slice to include.
        patch_shape (tuple[int, int]): Shape of the patches.
        patch_stride (int): Stride for patch creation.
        transform (Callable | None): Optional transformation to apply.
        volumetric_segments (list[VolumetricSegment]): Loaded volumetric segments.
        masks (list): Loaded masks for each segment.
        img_stacks (list): Loaded image stacks for each segment.
        usable_patch_position_arrs (np.ndarray): Usable patch positions.
        patch_pos_intervals (list): Patch position intervals.
    """

    def __init__(
        self,
        volumetric_segments: list[VolumetricSegment],
        z_min: int = 27,
        z_max: int = 37,
        patch_surface_shape: tuple[int, int] = (512, 512),
        patch_stride: int = 256,
        transform: Callable | None = None,
        prog_bar: bool = True,
    ):
        """Initialize the SurfaceVolumeDataset.

        Args:
            data_dir (Path): Directory containing the data.
            segment_ids (list[int]): List of segment IDs to load.
            z_min (int, optional): Minimum z-slice to include. Defaults to 27.
            z_max (int, optional): Maximum z-slice to include. Defaults to 37.
            patch_surface_shape (tuple[int, int], optional): Shape of the patches. Defaults to (512, 512).
            patch_stride (int, optional): Stride for patch creation. Defaults to 256.
            transform (Callable | None, optional): Optional transformation to apply. Defaults to None.
            prog_bar (bool, optional): Whether to show a progress bar while loading. Defaults to True.
        """
        self.transform = transform

        self.z_start = z_min
        self.z_end = z_max
        self.patch_shape = patch_surface_shape
        self.patch_stride = patch_stride

        self.volumetric_segments = volumetric_segments

        self.masks = [segment.load_mask() for segment in self.volumetric_segments]
        img_stack_iter = (
            tqdm(self.volumetric_segments, desc="Loading volumes...")
            if prog_bar
            else self.volumetric_segments
        )
        self.img_stacks = [
            segment.load_volume_as_memmap(z_start=self.z_start, z_end=self.z_end)
            for segment in img_stack_iter
        ]

        (
            self.usable_patch_position_arrs,
            original_patch_pos_sizes,
        ) = self.set_up_patch_positions()
        self.patch_pos_intervals = np.cumsum([0] + original_patch_pos_sizes).tolist()

    def __len__(self) -> int:
        """Get the number of usable patches.

        Returns:
            int: The number of usable patches in the dataset.
        """
        return self.usable_patch_position_arrs.shape[0]

    def create_usable_patch_map(self, fragment_idx: int) -> np.ndarray:
        """Create a usable patch map for a specific fragment.

        Args:
            fragment_idx (int): Index of the fragment to create the patch map for.

        Returns:
            np.ndarray: A boolean map indicating whether each patch is usable.
        """
        papyrus_mask = self.masks[fragment_idx]
        mask_patches = patchify(papyrus_mask, patch_size=self.patch_shape, step=self.patch_stride)
        # Create a usable patch map where true indicates that the patch is usable for training/validation/test and
        # false indicates there is not enough papyrus.
        usable_patch_map = np.empty(shape=(mask_patches.shape[:2]), dtype=bool)
        for i in range(mask_patches.shape[0]):
            for j in range(mask_patches.shape[1]):
                mask_patch = mask_patches[i, j]
                usable_patch_map[
                    i, j
                ] = mask_patch.any()  # A patch is "usable" if there are ANY papyrus pixels present.
        return usable_patch_map

    def create_usable_patch_position_arr(self, usable_patch_map: np.ndarray) -> np.ndarray:
        """Create an array of usable patch positions based on the patch map.

        Args:
            usable_patch_map (np.ndarray): A boolean map indicating patch usability.

        Returns:
            np.ndarray: An array of usable patch positions.
        """
        usable_patch_positions = []
        for i in range(usable_patch_map.shape[0]):
            for j in range(usable_patch_map.shape[1]):
                if usable_patch_map[i, j]:
                    position = patch_index_to_pixel_position(
                        i, j, self.patch_shape, self.patch_stride
                    )
                    usable_patch_positions.append(position)

        return np.array(usable_patch_positions)

    def set_up_patch_positions(self) -> tuple:
        """Set up the usable patch positions across all segments.

        Returns:
            tuple: A tuple containing an array of usable patch positions and a list of original patch position sizes.
        """
        usable_patch_position_arrs = []
        original_patch_pos_sizes = []
        for i in range(self.n_segments):
            usable_patch_map = self.create_usable_patch_map(i)
            usable_patch_position_arr = self.create_usable_patch_position_arr(usable_patch_map)
            original_patch_pos_sizes.append(usable_patch_position_arr.shape[0])
            usable_patch_position_arrs.append(usable_patch_position_arr)

        usable_patch_position_arrs = np.vstack(usable_patch_position_arrs)
        return usable_patch_position_arrs, original_patch_pos_sizes

    def idx_to_segment_idx(self, index: int) -> int:
        """Convert a global index to a segment index.

        Args:
            index (int): Global index in the dataset.

        Returns:
            int: Corresponding segment index.
        """
        return bisect.bisect_right(self.patch_pos_intervals, index) - 1

    @property
    def n_slices(self) -> int:
        """Get the number of surface volume layers used.

        Returns:
            int: The number of surface volume layers.
        """
        return self.z_end - self.z_start

    @property
    def segment_names(self) -> list[str]:
        return [segment.segment_name for segment in self.volumetric_segments]

    @property
    def n_segments(self) -> int:
        """Get the number of volumetric segments.

        Returns:
            int: The number of volumetric segments.
        """
        return len(self.segment_names)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(segment_names={self.segment_names})"
