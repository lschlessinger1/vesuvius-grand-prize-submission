import logging
from pathlib import Path

import albumentations as A
from albumentations.pytorch import ToTensorV2
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities import CombinedLoader
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
from torch.utils.data import DataLoader
from tqdm import tqdm

from vesuvius_challenge_rnd import FRAGMENT_DATA_DIR, SCROLL_DATA_DIR
from vesuvius_challenge_rnd.data import Fragment, Scroll, ScrollSegment
from vesuvius_challenge_rnd.data.preprocessors.fragment_preprocessor import FragmentPreprocessorBase
from vesuvius_challenge_rnd.fragment_ink_detection.ink_detection.data.base_fragment_data_module import (
    AbstractFragmentValPatchDataset,
)
from vesuvius_challenge_rnd.fragment_ink_detection.ink_detection.data.patch_dataset import (
    PatchDataset,
)
from vesuvius_challenge_rnd.scroll_ink_detection.ink_detection.data.scroll_patch_dataset import (
    ScrollPatchDataset,
)

ScrollSegmentsDataType = str | list[tuple[str, str]]


class CombinedDataModule(LightningDataModule, AbstractFragmentValPatchDataset):
    def __init__(
        self,
        train_fragment_ind: list[int],
        val_fragment_ind: list[int],
        train_scroll_segments: ScrollSegmentsDataType,
        val_scroll_segments: ScrollSegmentsDataType,
        fragment_data_dir: Path = FRAGMENT_DATA_DIR,
        scroll_data_dir: Path = SCROLL_DATA_DIR,
        fragment_batch_size: int = 2,
        scroll_batch_size: int = 2,
        z_min_fragment: int = 27,
        z_max_fragment: int = 37,
        z_min_scroll: int = 27,
        z_max_scroll: int = 37,
        patch_surface_shape: tuple[int, int] = (512, 512),
        patch_stride: int = 256,
        downsampling: int | None = None,
        num_workers: int | None = 0,
        slice_dropout_p: float = 0,
        non_destructive: bool = True,
        non_rigid: bool = False,
        fragment_preprocessor: FragmentPreprocessorBase | None = None,
    ):
        super().__init__()
        # Validate z_min and z_max.
        fragment_z_extent = z_max_fragment - z_min_fragment
        scroll_z_extent = z_max_scroll - z_min_scroll
        if fragment_z_extent != scroll_z_extent:
            raise ValueError(
                f"Fragment and scroll z-extents must be equal. Found fragment extent of {fragment_z_extent}"
                f" and scroll extent of {scroll_z_extent}."
            )

        self.fragment_data_dir = fragment_data_dir
        self.scroll_data_dir = scroll_data_dir
        self.batch_size_scroll = fragment_batch_size
        self.batch_size_fragment = scroll_batch_size
        self.train_fragment_ind = train_fragment_ind
        self.val_fragment_ind = val_fragment_ind
        self.train_scroll_segments = train_scroll_segments
        self.val_scroll_segments = val_scroll_segments
        self.z_min_fragment = z_min_fragment
        self.z_max_fragment = z_max_fragment
        self.z_min_scroll = z_min_scroll
        self.z_max_scroll = z_max_scroll
        self.patch_surface_shape = patch_surface_shape
        self.patch_stride = patch_stride
        self.downsampling = 1 if downsampling is None else downsampling
        self.num_workers = num_workers
        self.slice_dropout_p = slice_dropout_p
        self.non_destructive = non_destructive
        self.non_rigid = non_rigid
        self.fragment_preprocessor = fragment_preprocessor
        self.processed_fragment_data_dir = (
            self.fragment_data_dir
            if self.fragment_preprocessor is None
            else self.fragment_preprocessor.preprocessing_dir / FRAGMENT_DATA_DIR.name
        )

        # Create the scroll segments.
        self.segments_train = self._filter_segments(
            self._instantiate_segments(self.train_scroll_segments)
        )
        self.segments_val = self._filter_segments(
            self._instantiate_segments(self.val_scroll_segments)
        )

    def _instantiate_segments(
        self, scroll_segment_data: ScrollSegmentsDataType
    ) -> list[ScrollSegment]:
        if isinstance(scroll_segment_data, list):
            segments = []
            for scroll_id, segment_name in scroll_segment_data:
                segment = ScrollSegment(scroll_id, segment_name, scroll_dir=self.scroll_data_dir)
                segments.append(segment)
        elif scroll_segment_data in ("1", "2"):
            scroll = Scroll(scroll_id=scroll_segment_data, scroll_dir=self.scroll_data_dir)
            segments = scroll.segments
        elif scroll_segment_data.lower() == "all":
            scrolls = [Scroll(str(i + 1)) for i in range(2)]
            segments = scrolls[0].segments + scrolls[1].segments
        else:
            raise ValueError(
                "Scroll segments must be '1', '2', 'all', or a list of tuples of scroll IDs and segment names (e.g., "
                "[(1, 20230828154913), (1, 20230819210052)])."
            )

        return segments

    def _filter_segments(self, segments: list[ScrollSegment]):
        filtered_segments = []
        for segment in segments:
            if (
                segment.surface_shape[0] >= self.patch_surface_shape[0]
                and segment.surface_shape[1] >= self.patch_surface_shape[1]
            ):
                filtered_segments.append(segment)
            else:
                logging.warning(
                    f"Skipping scroll {segment.scroll_id} segment {segment.segment_name} with surface shape {segment.surface_shape} because "
                    f"it's smaller than the patch surface shape: {self.patch_surface_shape}."
                )
        return filtered_segments

    def prepare_data(self) -> None:
        super().prepare_data()
        if self.fragment_preprocessor is not None:
            all_fragment_ind = self.train_fragment_ind + self.val_fragment_ind
            fragments = [
                Fragment(fid, fragment_dir=self.fragment_data_dir) for fid in all_fragment_ind
            ]
            for fragment in tqdm(fragments, desc="Preprocessing fragments..."):
                self.fragment_preprocessor(fragment)

    def setup(self, stage: str) -> None:
        if stage == "fit" or stage is None:
            self.data_fragment_train = PatchDataset(
                self.processed_fragment_data_dir,
                self.train_fragment_ind,
                transform=self.train_transform_fragment(),
                patch_surface_shape=self.patch_surface_shape,
                patch_stride=self.patch_stride,
                z_min=self.z_min_fragment,
                z_max=self.z_max_fragment,
            )
            self.data_fragment_val = PatchDataset(
                self.processed_fragment_data_dir,
                self.val_fragment_ind,
                transform=self.validation_transform_fragment(),
                patch_surface_shape=self.patch_surface_shape,
                patch_stride=self.patch_stride,
                z_min=self.z_min_fragment,
                z_max=self.z_max_fragment,
            )
            self.data_scroll_train = ScrollPatchDataset(
                self.segments_train,
                transform=self.train_transform_scroll(),
                patch_surface_shape=self.patch_surface_shape,
                patch_stride=self.patch_stride,
                z_min=self.z_min_scroll,
                z_max=self.z_max_scroll,
            )
            self.data_scroll_val = ScrollPatchDataset(
                self.segments_val,
                transform=self.validation_transform_scroll(),
                patch_surface_shape=self.patch_surface_shape,
                patch_stride=self.patch_stride,
                z_min=self.z_min_scroll,
                z_max=self.z_max_scroll,
            )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        source_data_loader = DataLoader(
            self.data_fragment_train,
            batch_size=self.batch_size_scroll,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
        )
        target_data_loader = DataLoader(
            self.data_scroll_train,
            batch_size=self.batch_size_fragment,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
        )
        iterables = {
            "source": source_data_loader,
            "target": target_data_loader,
        }
        return CombinedLoader(iterables, mode="min_size")

    def val_dataloader(self) -> list[DataLoader]:
        source_data_loader = DataLoader(
            self.data_fragment_val,
            batch_size=self.batch_size_scroll,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )
        target_data_loader = DataLoader(
            self.data_scroll_val,
            batch_size=self.batch_size_fragment,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )
        return [source_data_loader, target_data_loader]

    def train_transform_fragment(self) -> A.Compose:
        """Define the transformations for the training stage.

        Returns:
            albumentations.Compose: A composed transformation object.
        """
        transforms = []

        if self.downsampling != 1:
            transforms += [A.Resize(self.patch_height, self.patch_width, always_apply=True)]

        if self.non_destructive:
            non_destructive_transformations = [  # Dihedral group D4
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),  # Randomly rotates by 0, 90, 180, 270 degrees
                A.Transpose(p=0.5),  # Switch X and Y axis.
            ]
            transforms += non_destructive_transformations

        if self.non_rigid:
            non_rigid_transformations = A.OneOf(
                [
                    A.GridDistortion(p=0.5),
                ],
                p=0.5,
            )
            transforms += non_rigid_transformations

        transforms += [A.ChannelDropout(channel_drop_range=(1, 2), p=self.slice_dropout_p)]
        transforms += [A.Normalize(mean=[0], std=[1]), ToTensorV2(transpose_mask=True)]

        return A.Compose(transforms)

    def validation_transform_fragment(self) -> A.Compose:
        """Define the transformations for the validation stage.

        Returns:
            albumentations.Compose: A composed transformation object.
        """
        transforms = []
        if self.downsampling != 1:
            transforms += [A.Resize(self.patch_height, self.patch_width, always_apply=True)]
        transforms += [A.Normalize(mean=[0], std=[1]), ToTensorV2(transpose_mask=True)]
        return A.Compose(transforms)

    def train_transform_scroll(self) -> A.Compose:
        transforms = []

        if self.downsampling != 1:
            transforms += [A.Resize(self.patch_height, self.patch_width, always_apply=True)]

        if self.non_destructive:
            non_destructive_transformations = [  # Dihedral group D4
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),  # Randomly rotates by 0, 90, 180, 270 degrees
                A.Transpose(p=0.5),  # Switch X and Y axis.
            ]
            transforms += non_destructive_transformations

        transforms += [A.Normalize(mean=[0], std=[1]), ToTensorV2(transpose_mask=True)]

        return A.Compose(transforms)

    def validation_transform_scroll(self) -> A.Compose:
        """Define the transformations for the validation stage.

        Returns:
            albumentations.Compose: A composed transformation object.
        """
        transforms = []
        if self.downsampling != 1:
            transforms += [A.Resize(self.patch_height, self.patch_width, always_apply=True)]
        transforms += [A.Normalize(mean=[0], std=[1]), ToTensorV2(transpose_mask=True)]
        return A.Compose(transforms)

    @property
    def patch_height(self) -> int:
        """The (possibly downsampled) patch height."""
        return self.patch_surface_shape[0] // self.downsampling

    @property
    def patch_width(self) -> int:
        """The (possibly downsampled) patch width."""
        return self.patch_surface_shape[1] // self.downsampling

    @property
    def patch_depth(self) -> int:
        """The patch depth."""
        return self.z_max_fragment - self.z_min_fragment

    @property
    def val_fragment_dataset(self) -> PatchDataset:
        """The fragment validation dataset."""
        return self.data_fragment_val
