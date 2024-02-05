from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

from vesuvius_challenge_rnd.data import Fragment, preprocess_subvolume
from vesuvius_challenge_rnd.fragment_ink_detection.ink_detection.data.in_memory_surface_volume_dataset import (
    SurfaceVolumeDataset,
)


class PatchDataset(SurfaceVolumeDataset):
    """A subclass of SurfaceVolumeDataset for handling patches of surface volume data.

    This class extends SurfaceVolumeDataset to include the loading and processing
    of patches, including ink labels.

    Attributes:
        data_dir (Path): Directory containing the data.
        fragment_ind (list[int]): List of fragment indices to load.
        z_min (int): Minimum z-slice to include.
        z_max (int): Maximum z-slice to include.
        patch_surface_shape (tuple[int, int]): Surface (height x width) shape of the patches.
        patch_stride (int): Stride for patch creation.
        transform (Callable | None): Optional transformation to apply.
        prog_bar (bool): Whether to show a progress bar while loading.
        labels (list): Loaded ink labels for each fragment.
    """

    def __init__(
        self,
        data_dir: Path,
        fragment_ind: list[int],
        z_min: int = 27,
        z_max: int = 37,
        patch_surface_shape: tuple[int, int] = (512, 512),
        patch_stride: int = 256,
        transform: Callable | None = None,
        prog_bar: bool = True,
    ):
        """Initialize the PatchDataset.

        Args:
            data_dir (Path): Directory containing the data.
            fragment_ind (list[int]): List of fragment indices to load.
            z_min (int, optional): Minimum z-slice to include. Defaults to 27.
            z_max (int, optional): Maximum z-slice to include. Defaults to 37.
            patch_surface_shape (tuple[int, int], optional): Shape of the patches. Defaults to (512, 512).
            patch_stride (int, optional): Stride for patch creation. Defaults to 256.
            transform (Callable | None, optional): Optional transformation to apply. Defaults to None.
            prog_bar (bool, optional): Whether to show a progress bar while loading. Defaults to True.
        """
        self._fragments = [Fragment(fid, fragment_dir=data_dir) for fid in fragment_ind]
        super().__init__(
            self._fragments,
            z_min,
            z_max,
            patch_surface_shape,
            patch_stride,
            transform,
            prog_bar,
        )
        self.data_dir = data_dir
        self.labels = [fragment.load_ink_labels() for fragment in self.fragments]

    def __getitem__(self, index: int) -> tuple:
        """Get a patch, corresponding label, and patch position by index.

        Args:
            index (int): Index of the patch to retrieve.

        Returns:
            tuple: A tuple containing the patch, label, and patch position.
        """
        fragment_idx = self.idx_to_segment_idx(index)

        patch_pos = self.usable_patch_position_arrs[index]
        ((y0, x0), (y1, x1)) = patch_pos

        patch = preprocess_subvolume(
            self.img_stacks[fragment_idx][:, y0:y1, x0:x1], slice_dim_last=True
        )

        patch_label = self.labels[fragment_idx][y0:y1, x0:x1].astype(np.uint8)

        if self.transform is not None:
            transformed = self.transform(image=patch, mask=patch_label)
            patch = transformed["image"]
            patch_label = transformed["mask"]

        return patch, patch_label, patch_pos

    @property
    def fragments(self) -> list[Fragment]:
        """Get the fragments.

        Returns:
            list[Fragment]: The list of fragments.
        """
        return self._fragments

    @property
    def fragment_ind(self) -> list[int]:
        """Get the fragment indices, alias for segment IDs.

        Returns:
            list[int]: List of fragment indices.
        """
        return self.segment_names

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(data_dir={self.data_dir}, fragment_ind={self.fragment_ind})"
        )
