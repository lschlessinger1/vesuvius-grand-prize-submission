from pathlib import Path

import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader

from vesuvius_challenge_rnd import FRAGMENT_DATA_DIR
from vesuvius_challenge_rnd.fragment_ink_detection.ink_detection.data.base_patch_data_module import (
    BasePatchDataModule,
)
from vesuvius_challenge_rnd.fragment_ink_detection.ink_detection.data.patch_dataset import (
    PatchDataset,
)


class EvalPatchDataModule(BasePatchDataModule):
    """A data module for handling evaluation and prediction of patches.

    This class extends BasePatchDataModule and includes functionality
    specific to the evaluation and prediction stages, such as dataset
    setup and transformation.

    Attributes:
        predict_fragment_ind (list[int]): List of fragment indices for prediction.
        data_predict (PatchDataset): Dataset for the prediction stage.
        data_dir (Path): Directory containing the fragment data. Defaults to FRAGMENT_DATA_DIR.
        z_min (int): Minimum z-slice to include. Defaults to 27.
        z_max (int): Maximum z-slice to include. Defaults to 37.
        patch_surface_shape (tuple[int, int]): Shape of the patches. Defaults to (512, 512).
        patch_stride (int): Stride for patch creation. Defaults to 256.
        downsampling (int | None): Downsampling factor, 1 if None. Defaults to None.
        num_workers (int): Number of workers for data loading. Defaults to 0.
        batch_size (int): Batch size for data loading. Defaults to 4.
    """

    def __init__(
        self,
        predict_fragment_ind: list[int],
        data_dir: Path = FRAGMENT_DATA_DIR,
        z_min: int = 27,
        z_max: int = 37,
        patch_surface_shape: tuple[int, int] = (512, 512),
        patch_stride: int = 256,
        downsampling: int | None = None,
        batch_size: int = 4,
        num_workers: int = 0,
    ):
        """Initialize the EvalPatchDataModule.

        Args:
            predict_fragment_ind (list[int]): List of fragment indices for prediction.
            data_dir (Path, optional): Directory containing the fragment data. Defaults to FRAGMENT_DATA_DIR.
            z_min (int, optional): Minimum z-slice to include. Defaults to 27.
            z_max (int, optional): Maximum z-slice to include. Defaults to 37.
            patch_surface_shape (tuple[int, int], optional): Shape of the patches. Defaults to (512, 512).
            patch_stride (int, optional): Stride for patch creation. Defaults to 256.
            downsampling (int | None, optional): Downsampling factor, 1 if None. Defaults to None.
            batch_size (int, optional): Batch size for data loading. Defaults to 4.
            num_workers (int, optional): Number of workers for data loading. Defaults to 0.
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
        self.predict_fragment_ind = predict_fragment_ind

    def setup(self, stage: str) -> None:
        """Set up the data for the given stage.

        Args:
            stage (str): The stage for which to set up the data (e.g., "predict").
        """
        if stage == "predict" or stage is None:
            self.data_predict = PatchDataset(
                self.data_dir,
                self.predict_fragment_ind,
                transform=self.predict_transform(),
                patch_surface_shape=self.patch_surface_shape,
                patch_stride=self.patch_stride,
                z_min=self.z_min,
                z_max=self.z_max,
            )

    def predict_transform(self) -> A.Compose:
        """Define the transformations for the prediction stage.

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

    def predict_dataloader(self) -> DataLoader:
        """Create a data loader for the prediction stage.

        Returns:
            DataLoader: A PyTorch DataLoader for prediction.
        """
        return DataLoader(
            self.data_predict,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )
