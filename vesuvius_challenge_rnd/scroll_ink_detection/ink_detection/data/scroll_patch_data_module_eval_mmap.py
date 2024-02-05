import logging
from pathlib import Path

import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
from torch.utils.data import DataLoader

from vesuvius_challenge_rnd import SCROLL_DATA_DIR
from vesuvius_challenge_rnd.data import ScrollSegment
from vesuvius_challenge_rnd.fragment_ink_detection.ink_detection.data.base_patch_data_module import (
    BasePatchDataModule,
)
from vesuvius_challenge_rnd.scroll_ink_detection.ink_detection.data.block_dataset_eval import (
    BlockZConstantDatasetEval,
)
from vesuvius_challenge_rnd.scroll_ink_detection.ink_detection.data.memmap import (
    MEMMAP_DIR,
    load_segment_as_memmap,
    save_segment_as_memmap,
    segment_to_memmap_path,
)
from vesuvius_challenge_rnd.scroll_ink_detection.ink_detection.data.patch_sampling import (
    get_all_patch_positions_non_masked_batched,
)
from vesuvius_challenge_rnd.scroll_ink_detection.ink_detection.data.scroll_patch_data_module import (
    compute_xy_padding,
    load_segment_mask,
)


class ScrollPatchDataModuleEvalMmap(BasePatchDataModule):
    def __init__(
        self,
        prediction_segment_id: str,
        scroll_id: str,
        data_dir: Path = SCROLL_DATA_DIR,
        z_min: int = 15,
        z_max: int = 45,
        size: int = 64,
        z_reverse: bool = False,
        tile_size: int = 256,
        patch_stride: int = 32,
        clip_min: int = 0,
        clip_max: int = 255,
        downsampling: int | None = None,
        batch_size: int = 512,
        num_workers: int = 0,
        load_in_memory: bool = True,
        memmap_dir: Path = MEMMAP_DIR,
        skip_save_mmap_if_exists: bool = True,
    ):
        super().__init__(
            data_dir=data_dir,
            z_min=z_min,
            z_max=z_max,
            patch_surface_shape=(size, size),
            patch_stride=patch_stride,
            downsampling=downsampling,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        self.size = size
        self.tile_size = tile_size
        self.z_reverse = z_reverse
        self.prediction_segment_id = prediction_segment_id
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.load_in_memory = load_in_memory
        self.memmap_dir = memmap_dir
        self.skip_save_mmap_if_exists = skip_save_mmap_if_exists

        # Instantiate segment.
        self.segment = ScrollSegment(
            scroll_id, self.prediction_segment_id, scroll_dir=self.data_dir
        )

    @property
    def z_extent(self) -> int:
        return self.z_max - self.z_min

    def prepare_data(self) -> None:
        """Download data to disk.

        This method checks if the data directory is empty and raises an exception
        if the data is not found.

        Raises:
            ValueError: If the data directory is empty.
        """
        super().prepare_data()

        self.memmap_dir.mkdir(exist_ok=True, parents=True)
        memmap_path = segment_to_memmap_path(self.segment, memmap_dir=self.memmap_dir)
        if memmap_path.exists() and self.skip_save_mmap_if_exists:
            logging.info(f"Found existing memmap for segment {self.segment.segment_name}.")
        else:
            logging.info(f"Creating new memmap for segment {self.segment.segment_name}.")
            save_segment_as_memmap(
                self.segment,
                memmap_path,
                load_in_memory=self.load_in_memory,
            )

    def setup(self, stage: str) -> None:
        """Set up the data for the given stage.

        Args:
            stage (str): The stage for which to set up the data (e.g., "predict").
        """
        if stage == "predict" or stage is None:
            mmap_path = segment_to_memmap_path(self.segment, memmap_dir=self.memmap_dir)
            img_stack_ref = load_segment_as_memmap(mmap_path)
            pad_y, pad_x = compute_xy_padding(
                img_shape=self.segment.surface_shape, tile_size=self.tile_size
            )
            segment_mask = load_segment_mask(self.segment, pad_y=pad_y, pad_x=pad_x)
            patch_positions = get_all_patch_positions_non_masked_batched(
                segment_mask, (self.size, self.size), self.patch_stride, chunk_size=self.tile_size
            )
            self.data_predict = BlockZConstantDatasetEval(
                img_stack_ref,
                np.array(patch_positions),
                self.size,
                z_start=self.z_min,
                z_extent=self.z_extent,
                transform=self.predict_transform(),
                z_reverse=self.z_reverse,
                padding_yx=(pad_y, pad_x),
                clip_min=self.clip_min,
                clip_max=self.clip_max,
            )

    def predict_transform(self) -> A.Compose:
        return A.Compose(
            [
                A.Resize(self.size, self.size),
                A.Normalize(mean=[0] * self.z_extent, std=[1] * self.z_extent),
                ToTensorV2(transpose_mask=True),
            ]
        )

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.data_predict,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
        )
