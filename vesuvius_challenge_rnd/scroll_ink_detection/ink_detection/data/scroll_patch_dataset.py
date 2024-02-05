from collections.abc import Callable

from vesuvius_challenge_rnd.data import preprocess_subvolume
from vesuvius_challenge_rnd.data.volumetric_segment import VolumetricSegment
from vesuvius_challenge_rnd.fragment_ink_detection.ink_detection.data.in_memory_surface_volume_dataset import (
    SurfaceVolumeDataset,
)


class ScrollPatchDataset(SurfaceVolumeDataset):
    def __init__(
        self,
        segments: list[VolumetricSegment],
        z_min: int = 27,
        z_max: int = 37,
        patch_surface_shape: tuple[int, int] = (512, 512),
        patch_stride: int = 256,
        transform: Callable | None = None,
        prog_bar: bool = True,
    ):
        """Initialize the PatchDataset.

        Args:
            segments (list[int]): List of segment IDs to load.
            z_min (int, optional): Minimum z-slice to include. Defaults to 27.
            z_max (int, optional): Maximum z-slice to include. Defaults to 37.
            patch_surface_shape (tuple[int, int], optional): Shape of the patches. Defaults to (512, 512).
            patch_stride (int, optional): Stride for patch creation. Defaults to 256.
            transform (Callable | None, optional): Optional transformation to apply. Defaults to None.
            prog_bar (bool, optional): Whether to show a progress bar while loading. Defaults to True.
        """
        super().__init__(
            segments,
            z_min,
            z_max,
            patch_surface_shape,
            patch_stride,
            transform,
            prog_bar,
        )

    def __getitem__(self, index: int) -> tuple:
        """Get a patch and patch position by index.

        Args:
            index (int): Index of the patch to retrieve.

        Returns:
            tuple: A tuple containing the patch and patch position.
        """
        fragment_idx = self.idx_to_segment_idx(index)

        patch_pos = self.usable_patch_position_arrs[index]
        ((y0, x0), (y1, x1)) = patch_pos

        patch = preprocess_subvolume(
            self.img_stacks[fragment_idx][:, y0:y1, x0:x1], slice_dim_last=True
        )

        if self.transform is not None:
            transformed = self.transform(image=patch)
            patch = transformed["image"]

        return patch, patch_pos
