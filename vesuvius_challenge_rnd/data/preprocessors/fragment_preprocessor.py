import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path

import cv2
import numpy as np
from tifffile import tifffile

from vesuvius_challenge_rnd import DATA_DIR
from vesuvius_challenge_rnd.data import Fragment
from vesuvius_challenge_rnd.data.fragment import VOXEL_SIZE_MICRONS as FRAGMENT_VOXEL_SIZE_MICRONS
from vesuvius_challenge_rnd.data.scroll import VOXEL_SIZE_MICRONS as SCROLL_VOXEL_SIZE_MICRONS
from vesuvius_challenge_rnd.data.util import get_img_width_height

DEFAULT_ZOOM_FACTOR = FRAGMENT_VOXEL_SIZE_MICRONS / SCROLL_VOXEL_SIZE_MICRONS
DEFAULT_OUTPUT_DIR = DATA_DIR / "processed"


class FragmentPreprocessorBase(ABC):
    """A class that preprocesses surface volumes, masks, ink labels, and optionally IR images."""

    def __init__(
        self,
        output_dir: Path = DEFAULT_OUTPUT_DIR,
        preprocess_ir_img: bool = False,
        skip_if_exists: bool = True,
        check_img_shapes: bool = True,
    ):
        self.output_dir = output_dir
        self.preprocess_ir_img = preprocess_ir_img
        self.skip_if_exists = skip_if_exists
        self.check_img_shapes = check_img_shapes

    def __call__(self, fragment: Fragment) -> None:
        """Preprocess surface volumes, mask, ink labels, IR image (optional), and save a param file."""
        output_volume_dir = self._get_new_volume_dir(fragment)
        new_surface_volume_dir = output_volume_dir / fragment.surface_volume_dir_name
        if self.skip_if_exists and new_surface_volume_dir.exists():
            logging.info(
                "Skipping surface volume preprocessing because the directory already exists."
            )
        else:
            self._preprocess_surface_volumes(fragment, new_surface_volume_dir)

        output_mask_path = output_volume_dir / fragment.papyrus_mask_file_name
        if self.skip_if_exists and output_mask_path.exists():
            logging.info("Skipping mask preprocessing because it already exists.")
        else:
            self._preprocess_mask(fragment, output_mask_path)

        output_ink_labels_path = output_volume_dir / fragment.ink_labels_path.name
        if self.skip_if_exists and output_ink_labels_path.exists():
            logging.info("Skipping ink labels preprocessing because it already exists.")
        else:
            self._preprocess_labels(fragment, output_ink_labels_path)

        if self.preprocess_ir_img:
            output_ir_img_path = output_volume_dir / fragment.ir_img_path.name
            if self.skip_if_exists and output_ir_img_path.exists():
                logging.info("Skipping IR image preprocessing because it already exists.")
            else:
                self._preprocess_ir_img(fragment, output_ir_img_path)
        else:
            output_ir_img_path = None

        output_json_path = output_volume_dir / "preprocessing_params.json"
        if self.skip_if_exists and output_json_path.exists():
            logging.info("Skipping param saving because it already exists.")
        else:
            self._save_params(output_json_path)

        if self.check_img_shapes:
            # Check if all image shapes are the same.
            self._verify_img_surface_shapes(
                new_surface_volume_dir, output_mask_path, output_ink_labels_path, output_ir_img_path
            )

    @abstractmethod
    def _preprocess_surface_volumes(self, fragment: Fragment, output_dir: Path) -> None:
        raise NotImplementedError

    @abstractmethod
    def _preprocess_mask(self, fragment: Fragment, output_path: Path) -> None:
        raise NotImplementedError

    @abstractmethod
    def _preprocess_labels(self, fragment: Fragment, output_path: Path) -> None:
        raise NotImplementedError

    @abstractmethod
    def _preprocess_ir_img(self, fragment: Fragment, output_path: Path) -> None:
        raise NotImplementedError

    @property
    @abstractmethod
    def method_name(self) -> str:
        raise NotImplementedError

    def _save_surface_volumes(self, surface_volumes: np.ndarray, output_dir: Path) -> None:
        """Save the surface volumes to the preprocessing directory."""
        _verify_is_dtype(surface_volumes, dtype=np.uint16)
        output_dir.mkdir(exist_ok=True, parents=True)

        for i, img in enumerate(surface_volumes):
            output_path = output_dir / f"{str(i).zfill(2)}.tif"
            tifffile.imwrite(output_path, img)

        logging.debug(f"Saved {len(surface_volumes)} surface volumes.")

    def _save_mask(self, mask: np.ndarray, output_path: Path) -> None:
        """Save the papyrus mask to the preprocessing directory."""
        _imwrite_uint8(mask, output_path)
        logging.debug(f"Saved mask of shape {mask.shape}.")

    def _save_ink_labels(self, ink_labels: np.ndarray, output_path: Path) -> None:
        """Save the ink labels to the preprocessing directory."""
        _imwrite_uint8(ink_labels, output_path)
        logging.debug(f"Saved ink labels of shape {ink_labels.shape}.")

    def _save_ir_img(self, ir_img: np.ndarray, output_path: Path) -> None:
        """Save the IR image to the preprocessing directory."""
        _imwrite_uint8(ir_img, output_path)
        logging.debug(f"Saved IR image of shape {ir_img.shape}.")

    @property
    def preprocessing_dir(self):
        return self.output_dir / self.method_name

    def _get_new_volume_dir(self, fragment: Fragment) -> Path:
        return self.preprocessing_dir / fragment.data_dir.name / fragment.volume_dir_path.name

    def _save_params(self, output_json_path: Path):
        # Collect parameters in a dictionary
        params = {key: value for key, value in self.__dict__.items() if not key.startswith("_")}

        # Convert Path objects to strings
        for key, value in params.items():
            if isinstance(value, Path):
                params[key] = str(value)

        # Serialize and save the dictionary as a JSON file
        with open(output_json_path, "w") as json_file:
            json.dump(params, json_file)

        logging.debug(f"Saved preprocessing params to {output_json_path}.")

    @staticmethod
    def _verify_img_surface_shapes(
        new_surface_volume_dir: Path,
        output_mask_path: Path,
        output_ink_labels_path: Path,
        output_ir_img_path: Path,
    ) -> None:
        shapes = {}
        surface_vol_0_path = list(new_surface_volume_dir.glob("*.tif"))[0]
        surface_width_height = get_img_width_height(surface_vol_0_path)
        shapes["surface_vol_0"] = surface_width_height

        mask_width_height = get_img_width_height(output_mask_path)
        shapes["mask"] = mask_width_height

        labels_width_height = get_img_width_height(output_ink_labels_path)
        shapes["ink_labels"] = labels_width_height

        if output_ir_img_path is not None:
            ir_width_height = get_img_width_height(output_ir_img_path)
            shapes["ir_img"] = ir_width_height

        shapes_equal = _check_shapes_equal(list(shapes.values()))
        if not shapes_equal:
            logging.warning(f"Found unequal image shapes (width, height): {shapes}")

    def __repr__(self):
        return f"{type(self).__name__}(output_dir={self.output_dir}, preprocess_ir_img={self.preprocess_ir_img})"


class FragmentPreprocessor(FragmentPreprocessorBase):
    def _preprocess_surface_volumes(self, fragment: Fragment, output_dir: Path) -> None:
        """Load, transform, and save surface volumes."""
        surface_volumes = fragment.load_volume_as_memmap()
        surface_volumes = self._transform_surface_volumes(surface_volumes)
        self._save_surface_volumes(surface_volumes, output_dir)

    def _preprocess_mask(self, fragment: Fragment, output_path: Path) -> None:
        """Load, transform, and save the papyrus mask."""
        mask = fragment.load_mask()
        mask = self._transform_mask(mask)
        self._save_mask(mask, output_path)

    def _preprocess_labels(self, fragment: Fragment, output_path: Path) -> None:
        """Load, transform, and save the ink labels."""
        labels = fragment.load_ink_labels()
        labels = self._transform_labels(labels)
        self._save_ink_labels(labels, output_path)

    def _preprocess_ir_img(self, fragment: Fragment, output_path: Path) -> None:
        """Load, transform, and save the infrared image."""
        ir_img = fragment.load_ir_img()
        ir_img = self._transform_ir_img(ir_img)
        self._save_ir_img(ir_img, output_path)

    @abstractmethod
    def _transform_surface_volumes(self, surface_volumes: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def _transform_labels(self, labels: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def _transform_mask(self, mask: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _transform_ir_img(self, ir_img: np.ndarray) -> np.ndarray:
        return ir_img


def _imwrite_uint8(array: np.ndarray, output_path: str | Path):
    _verify_is_dtype(array, dtype=np.uint8)
    cv2.imwrite(str(output_path), array)


def _verify_is_dtype(arr: np.ndarray, dtype: type[np.dtype]) -> None:
    """
    Verifies if the input NumPy array is of a specific dtype.

    Args:
        arr (ndarray): The input NumPy array.
        dtype (type[np.dtype]): The expected data type.

    Returns:
        None: Returns nothing if dtype matches the expected dtype.

    Raises:
        ValueError: If dtype of the input array does not match the expected dtype.
    """
    if arr.dtype != dtype:
        raise ValueError(f"Input array must be of dtype {dtype}.")
    return None


def _check_shapes_equal(shapes: list[tuple[int, ...]]) -> bool:
    """
    Check if all shapes in the list are equal.

    Args:
        shapes (List[tuple[int, ...]]): A list of shapes, each represented as a tuple of ints.

    Returns:
        bool: True if all shapes are equal, False otherwise.

    Example:
        >>> _check_shapes_equal([(600, 800, 3), (600, 800, 3), (600, 800, 3)])
        True
        >>> _check_shapes_equal([(600, 800, 3), (600, 800, 4)])
        False
    """
    return len(set(shapes)) == 1 if shapes else False
