from pathlib import Path

import pytorch_lightning as pl

from vesuvius_challenge_rnd import FRAGMENT_DATA_DIR


class BasePatchDataModule(pl.LightningDataModule):
    """A base data module for handling patches of fragment data.

    This class provides a foundational structure for data loading and preparation
    in the context of PyTorch Lightning. It can be used as a base class for more
    specific data modules for different tasks.

    Attributes:
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
        data_dir: Path = FRAGMENT_DATA_DIR,
        z_min: int = 27,
        z_max: int = 37,
        patch_surface_shape: tuple[int, int] = (512, 512),
        patch_stride: int = 256,
        downsampling: int | None = None,
        batch_size: int = 4,
        num_workers: int = 0,
    ):
        """Initialize the BasePatchDataModule.

        Args:
            data_dir (Path, optional): Directory containing the fragment data. Defaults to FRAGMENT_DATA_DIR.
            z_min (int, optional): Minimum z-slice to include. Defaults to 27.
            z_max (int, optional): Maximum z-slice to include. Defaults to 37.
            patch_surface_shape (tuple[int, int], optional): Shape of the patches. Defaults to (512, 512).
            patch_stride (int, optional): Stride for patch creation. Defaults to 256.
            downsampling (int | None, optional): Downsampling factor, 1 if None. Defaults to None.
            batch_size (int, optional): Batch size for data loading. Defaults to 4.
            num_workers (int, optional): Number of workers for data loading. Defaults to 0.
        """
        super().__init__()
        self.data_dir = data_dir
        self.z_min = z_min
        self.z_max = z_max
        self.patch_surface_shape = patch_surface_shape
        self.patch_stride = patch_stride
        self.downsampling = 1 if downsampling is None else downsampling

        self.num_workers = num_workers
        self.batch_size = batch_size

    def prepare_data(self) -> None:
        """Download data to disk.

        This method checks if the data directory is empty and raises an exception
        if the fragment data is not found.

        Raises:
            ValueError: If the data directory is empty.
        """
        data_dir_is_empty = not any(self.data_dir.iterdir())
        if data_dir_is_empty:
            raise ValueError(
                f"Data directory ({self.data_dir}) is empty. Please download the data."
            )
