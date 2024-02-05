from pathlib import Path

import numpy as np
from tqdm import tqdm
from wand.image import Image

from vesuvius_challenge_rnd.data import Fragment
from vesuvius_challenge_rnd.data.preprocessors.fragment_preprocessor import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_ZOOM_FACTOR,
    FragmentPreprocessorBase,
)

DEFAULT_INT_ZOOM_FACTOR = int(1 / DEFAULT_ZOOM_FACTOR)


class InkIdPreprocessor(FragmentPreprocessorBase):
    def __init__(
        self,
        output_dir: Path = DEFAULT_OUTPUT_DIR,
        zoom_factor: int = DEFAULT_INT_ZOOM_FACTOR,
        upsample: bool = False,
        pbar: bool = True,
        preprocess_ir_img: bool = False,
        skip_if_exists: bool = True,
    ):
        super().__init__(
            output_dir, preprocess_ir_img=preprocess_ir_img, skip_if_exists=skip_if_exists
        )
        self.zoom_factor = zoom_factor
        self.upsample = upsample
        self.pbar = pbar

    def _preprocess_surface_volumes(self, fragment: Fragment, output_dir: Path) -> None:
        paths = fragment.all_surface_volume_paths
        output_volume = apply_inkid_resolution_matching(
            paths, int_zoom_factor=self.zoom_factor, upsample=self.upsample, pbar=self.pbar
        )

        # Reshape the array to D x H x W.
        output_volume = output_volume.transpose((2, 0, 1))

        # Rescale to int16 range from uint8 range.
        output_volume = np.iinfo(np.uint16).max * (output_volume / np.iinfo(np.uint8).max)
        output_volume = output_volume.astype(np.uint16)

        # Save output volume.
        self._save_surface_volumes(output_volume, output_dir)

    def _preprocess_mask(self, fragment: Fragment, output_path: Path) -> None:
        height, width, _ = compute_img_shape(
            fragment.all_surface_volume_paths, zoom_factor=self.zoom_factor, upsample=self.upsample
        )
        mask = wand_resize_to_array(
            fragment.papyrus_mask_path, width, height, blob_format="GRAY", dtype=np.uint8
        )
        self._save_mask(mask, output_path)

    def _preprocess_labels(self, fragment: Fragment, output_path: Path) -> None:
        height, width, _ = compute_img_shape(
            fragment.all_surface_volume_paths, zoom_factor=self.zoom_factor, upsample=self.upsample
        )
        ink_labels = wand_resize_to_array(
            fragment.ink_labels_path, width, height, blob_format="GRAY", dtype=np.uint8
        )
        self._save_ink_labels(ink_labels, output_path)

    def _preprocess_ir_img(self, fragment: Fragment, output_path: Path) -> None:
        height, width, _ = compute_img_shape(
            fragment.all_surface_volume_paths, zoom_factor=self.zoom_factor, upsample=self.upsample
        )
        ir_img = wand_resize_to_array(
            fragment.ir_img_path, width, height, blob_format="GRAY", dtype=np.uint8
        )
        self._save_ir_img(ir_img, output_path)

    @property
    def method_name(self) -> str:
        return f"ink-id__zoom={self.zoom_factor}__upsample={self.upsample}"

    def __repr__(self):
        return (
            f"{type(self).__name__}(output_dir={self.output_dir}, preprocess_ir_img={self.preprocess_ir_img}, "
            f"zoom_factor={self.zoom_factor}, upsample={self.upsample})"
        )


def wand_to_array(
    image_wand: Image, blob_format: str | None = None, dtype: type[np.dtype] | None = None
) -> np.ndarray:
    width, height = image_wand.size
    blob = image_wand.make_blob(format=blob_format)
    array = np.frombuffer(blob, dtype=dtype)
    array = array.reshape(height, width)
    return array


def wand_resize_to_array(
    img_path: Path,
    width: int,
    height: int,
    blob_format: str | None = None,
    dtype: type[np.dtype] | None = None,
) -> np.ndarray:
    with Image(filename=img_path) as img:
        with img.clone() as img_clone:
            img_clone.resize(width=width, height=height)
            array = wand_to_array(img_clone, blob_format=blob_format, dtype=dtype)
    return array


def compute_img_shape(
    image_filenames: list[Path], zoom_factor: int = 2, upsample: bool = False
) -> tuple[int, int, int]:
    with Image(filename=image_filenames[0]) as img:
        original_height = img.height
        original_width = img.width
        n_slices = len(image_filenames)
        if not upsample:
            new_height = original_height // zoom_factor
            new_width = original_width // zoom_factor
            return new_height, new_width, n_slices
        else:
            return original_height, original_width, n_slices


def apply_inkid_resolution_matching(
    image_filenames: list[Path], int_zoom_factor: int = 2, upsample: bool = False, pbar: bool = True
) -> np.ndarray:
    # Remove slices to downsample on z-axis.
    image_filenames = image_filenames[::int_zoom_factor]

    height, width, n_slices = compute_img_shape(
        image_filenames, zoom_factor=int_zoom_factor, upsample=upsample
    )
    output_volume = np.empty((height, width, n_slices), dtype=np.uint8)

    # Use ImageMagick to downsample remaining images along the x and y-axes.
    enumerable = tqdm(image_filenames, "Resizing images...") if pbar else image_filenames
    for i, image_filename in enumerate(enumerable):
        with Image(filename=image_filename) as img:
            with img.clone() as img_clone:
                img_clone.resize(width=width, height=height)
                output_volume[:, :, i] = np.array(img_clone, dtype=np.uint8).squeeze()

    return output_volume
