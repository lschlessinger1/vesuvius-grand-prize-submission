from pathlib import Path

import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from tqdm import tqdm

from vesuvius_challenge_rnd import FRAGMENT_DATA_DIR
from vesuvius_challenge_rnd.data import Fragment
from vesuvius_challenge_rnd.data.preprocessors.fragment_preprocessor import FragmentPreprocessorBase
from vesuvius_challenge_rnd.fragment_ink_detection.ink_detection.data.base_fragment_data_module import (
    AbstractFragmentValPatchDataset,
)
from vesuvius_challenge_rnd.fragment_ink_detection.ink_detection.data.base_patch_data_module import (
    BasePatchDataModule,
)
from vesuvius_challenge_rnd.fragment_ink_detection.ink_detection.data.patch_dataset import (
    PatchDataset,
)


class PatchDataModule(BasePatchDataModule, AbstractFragmentValPatchDataset):
    """A data module for handling training and validation of patches.

    This class extends BasePatchDataModule and includes functionality
    specific to the training and validation stages, such as dataset
    setup and transformation.

    Attributes:
        train_fragment_ind (list[int]): List of fragment indices for training.
        val_fragment_ind (list[int]): List of fragment indices for validation.
        data_train (PatchDataset): Dataset for the training stage.
        data_val (PatchDataset): Dataset for the validation stage.
        data_dir (Path): Directory containing the fragment data. Defaults to FRAGMENT_DATA_DIR.
        z_min (int): Minimum z-slice to include. Defaults to 27.
        z_max (int): Maximum z-slice to include. Defaults to 37.
        patch_surface_shape (tuple[int, int]): Shape of the patches. Defaults to (512, 512).
        patch_stride (int): Stride for patch creation. Defaults to 256.
        downsampling (int | None): Downsampling factor, 1 if None. Defaults to None.
        num_workers (int | None): Number of workers for data loading. Defaults to 0.
        batch_size (int): Batch size for data loading. Defaults to 4.
    """

    def __init__(
        self,
        train_fragment_ind: list[int],
        val_fragment_ind: list[int],
        data_dir: Path = FRAGMENT_DATA_DIR,
        z_min: int = 27,
        z_max: int = 37,
        patch_surface_shape: tuple[int, int] = (512, 512),
        patch_stride: int = 256,
        downsampling: int | None = None,
        batch_size: int = 4,
        num_workers: int | None = 0,
        slice_dropout_p: float = 0,
        non_destructive: bool = True,
        non_rigid: bool = True,
        fragment_preprocessor: FragmentPreprocessorBase | None = None,
    ):
        """Initialize the PatchDataModule.

        Args:
            train_fragment_ind (list[int]): List of fragment indices for training.
            val_fragment_ind (list[int]): List of fragment indices for validation.
            data_dir (Path, optional): Directory containing the fragment data. Defaults to FRAGMENT_DATA_DIR.
            z_min (int, optional): Minimum z-slice to include. Defaults to 27.
            z_max (int, optional): Maximum z-slice to include. Defaults to 37.
            patch_surface_shape (tuple[int, int], optional): Shape of the patches. Defaults to (512, 512).
            patch_stride (int, optional): Stride for patch creation. Defaults to 256.
            downsampling (int | None, optional): Downsampling factor, 1 if None. Defaults to None.
            batch_size (int, optional): Batch size for data loading. Defaults to 4.
            num_workers (int | None, optional): Number of workers for data loading. Defaults to 0.
            slice_dropout_p (int, optional): The probability of applying slice dropout. Defaults to 0.
            non_destructive (bool, optional): Apply non-destructive transformations. Defaults to True.
            non_rigid (bool, optional): Apply non-rigid transformations. Defaults to False.
            fragment_preprocessor (FragmentPreprocessorBase, optional): a fragment preprocessor. Defaults to None.

        """
        super().__init__(
            data_dir,
            z_min,
            z_max,
            patch_surface_shape,
            patch_stride,
            downsampling,
            batch_size,
            num_workers,
        )
        self.train_fragment_ind = train_fragment_ind
        self.val_fragment_ind = val_fragment_ind
        self.non_destructive = non_destructive
        self.non_rigid = non_rigid
        self.slice_dropout_p = slice_dropout_p
        self.fragment_preprocessor = fragment_preprocessor
        self.processed_data_dir = (
            self.data_dir
            if self.fragment_preprocessor is None
            else self.fragment_preprocessor.preprocessing_dir / FRAGMENT_DATA_DIR.name
        )

    def prepare_data(self) -> None:
        super().prepare_data()
        if self.fragment_preprocessor is not None:
            all_fragment_ind = self.train_fragment_ind + self.val_fragment_ind
            fragments = [Fragment(fid, fragment_dir=self.data_dir) for fid in all_fragment_ind]
            for fragment in tqdm(fragments, desc="Preprocessing fragments..."):
                self.fragment_preprocessor(fragment)

    def setup(self, stage: str) -> None:
        """Set up the data for the given stage.

        Args:
            stage (str): The stage for which to set up the data (e.g., "fit").
        """
        if stage == "fit" or stage is None:
            self.data_train = PatchDataset(
                self.processed_data_dir,
                self.train_fragment_ind,
                transform=self.train_transform(),
                patch_surface_shape=self.patch_surface_shape,
                patch_stride=self.patch_stride,
                z_min=self.z_min,
                z_max=self.z_max,
            )

            self.data_val = PatchDataset(
                self.processed_data_dir,
                self.val_fragment_ind,
                transform=self.validation_transform(),
                patch_surface_shape=self.patch_surface_shape,
                patch_stride=self.patch_stride,
                z_min=self.z_min,
                z_max=self.z_max,
            )

    def train_transform(self) -> A.Compose:
        """Define the transformations for the training stage.

        Returns:
            albumentations.Compose: A composed transformation object.
        """
        transforms = []
        height = self.patch_surface_shape[0] // self.downsampling
        width = self.patch_surface_shape[1] // self.downsampling
        if self.downsampling != 1:
            transforms += [A.Resize(height, width, always_apply=True)]

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
                    A.MotionBlur(p=0.5),
                    A.OpticalDistortion(p=0.5),
                ],
                p=0.5,
            )
            transforms += non_rigid_transformations

        transforms += [
            A.ChannelDropout(channel_drop_range=(1, 2), p=self.slice_dropout_p),
            A.RandomGamma(p=0.5),
            A.Normalize(mean=[0], std=[1]),
            ToTensorV2(transpose_mask=True),
        ]

        return A.Compose(transforms)

    def validation_transform(self) -> A.Compose:
        """Define the transformations for the validation stage.

        Returns:
            albumentations.Compose: A composed transformation object.
        """
        transforms = []
        if self.downsampling != 1:
            height = self.patch_surface_shape[0] // self.downsampling
            width = self.patch_surface_shape[1] // self.downsampling
            transforms += [A.Resize(height, width, always_apply=True)]
        transforms += [A.Normalize(mean=[0], std=[1]), ToTensorV2(transpose_mask=True)]
        return A.Compose(transforms)

    def train_dataloader(self) -> DataLoader:
        """Create a data loader for the training stage.

        Returns:
            DataLoader: A PyTorch DataLoader for training.
        """
        return DataLoader(
            self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Create a data loader for the validation stage.

        Returns:
            DataLoader: A PyTorch DataLoader for validation.
        """
        return DataLoader(
            self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )

    @property
    def val_fragment_dataset(self) -> PatchDataset:
        return self.data_val
