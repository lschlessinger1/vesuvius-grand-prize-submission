import os
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
from albumentations.pytorch import ToTensorV2
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
from torch.utils.data import DataLoader
from tqdm import tqdm

from vesuvius_challenge_rnd import SCROLL_DATA_DIR
from vesuvius_challenge_rnd.fragment_ink_detection.ink_detection.data.base_patch_data_module import (
    BasePatchDataModule,
)
from vesuvius_challenge_rnd.scroll_ink_detection.ink_detection.data.younader_scroll_patch_dataset_eval import (
    CustomDatasetTest,
)


class ScrollPatchDataModuleEval(BasePatchDataModule):
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
        downsampling: int | None = None,
        batch_size: int = 512,
        num_workers: int = 0,
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
        self.segment_dir = self.data_dir / scroll_id

    @property
    def z_extent(self) -> int:
        return self.z_max - self.z_min

    def setup(self, stage: str) -> None:
        """Set up the data for the given stage.

        Args:
            stage (str): The stage for which to set up the data (e.g., "predict").
        """
        if stage == "predict" or stage is None:
            images, xyxys = get_img_splits(
                self.prediction_segment_id,
                str(self.segment_dir),
                self.z_reverse,
                self.tile_size,
                self.patch_stride,
                self.z_min,
                self.z_max,
            )
            if len(xyxys) == 0:
                raise ValueError("No patches found.")

            self.data_predict = CustomDatasetTest(
                images,
                np.stack(xyxys),
                transform=self.predict_transform(),
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


def read_image_mask(
    segment_id: str,
    segment_path: str,
    reverse: bool,
    tile_size: int,
    start_idx: int = 15,
    end_idx: int = 45,
):
    images = []

    idxs = range(start_idx, end_idx)
    is_monster_segment = segment_id.startswith("Scroll1_part_1_wrap")
    for i in idxs:
        surface_vol_dirname = "layers" if not is_monster_segment else "surface_volume"
        surface_vol_dir = Path(segment_path) / segment_id / surface_vol_dirname
        n_digits = len(list(surface_vol_dir.glob("*.tif"))[0].stem)
        tif_num = str(i).zfill(n_digits)
        tif_path = surface_vol_dir / f"{tif_num}.tif"
        image = cv2.imread(str(tif_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"image is None ({segment_id} - i={i})")
        pad0 = tile_size - image.shape[0] % tile_size
        pad1 = tile_size - image.shape[1] % tile_size

        image = np.pad(image, [(0, pad0), (0, pad1)], constant_values=0)
        image = np.clip(image, 0, 255)

        images.append(image)
    images = np.stack(images, axis=2)
    if reverse:
        images = images[:, :, ::-1]
    segment_mask = None
    if os.path.exists(f"{segment_path}/{segment_id}/{segment_id}_mask.png"):
        segment_mask = cv2.imread(f"{segment_path}/{segment_id}/{segment_id}_mask.png", 0)
        segment_mask = np.pad(segment_mask, [(0, pad0), (0, pad1)], constant_values=0)
        kernel = np.ones((16, 16), np.uint8)
        segment_mask = cv2.erode(segment_mask, kernel, iterations=1)
    return images, segment_mask


def get_img_splits(
    segment_id: str,
    segment_path: str,
    reverse: bool,
    tile_size: int,
    stride: int,
    start_idx: int,
    end_idx: int,
):
    images = []
    xyxys = []
    image, segment_mask = read_image_mask(
        segment_id, segment_path, reverse, tile_size, start_idx, end_idx
    )

    if image is None:
        raise ValueError(f"image is None (segment={segment_id})")

    if segment_mask is None:
        raise ValueError(f"segment_mask is None (segment={segment_id})")

    x1_list = list(range(0, image.shape[1] - tile_size + 1, stride))
    y1_list = list(range(0, image.shape[0] - tile_size + 1, stride))
    for y1 in tqdm(y1_list, desc="Getting img splits..."):
        for x1 in x1_list:
            y2 = y1 + tile_size
            x2 = x1 + tile_size
            # Must be fully within the masked region.
            if not np.any(segment_mask[y1:y2, x1:x2] == 0):
                images.append(image[y1:y2, x1:x2])
                xyxys.append([x1, y1, x2, y2])
    return images, xyxys
