from pathlib import Path

import numpy as np
import pint
from PIL import Image

from vesuvius_challenge_rnd import FRAGMENT_DATA_DIR, ureg
from vesuvius_challenge_rnd.data.volumetric_segment import VolumetricSegment

VOXEL_SIZE_MICRONS = 3.24
VOXEL_SIZE = VOXEL_SIZE_MICRONS * ureg.micron


class Fragment(VolumetricSegment):
    """A fragment of a scroll.

    The Fragment class represents a specific segment or fragment of a scroll. It inherits from
    the VolumetricSegment class and defines additional properties and methods specific to the
    fragment, such as paths to ink labels and infrared (IR) images.
    """

    def __init__(self, fragment_id: int, fragment_dir: Path = FRAGMENT_DATA_DIR):
        """
        Initializes a Fragment instance.

        Args:
            fragment_id (int): The unique identifier for the fragment.
            fragment_dir (Path, optional): The directory where the fragment data is located.
                Defaults to FRAGMENT_DATA_DIR.
        """
        super().__init__(data_dir=fragment_dir, segment_name=str(fragment_id))
        self.fragment_id = fragment_id

    @property
    def ink_labels_path(self) -> Path:
        """Path to the ink labels file of the fragment.

        Returns:
            Path: The file path to the ink labels.
        """
        return self.volume_dir_path / "inklabels.png"

    @property
    def ir_img_path(self):
        """Path to the infrared (IR) image file of the fragment.

        Returns:
            Path: The file path to the IR image.
        """
        return self.volume_dir_path / "ir.png"

    @property
    def papyrus_mask_file_name(self) -> str:
        """File name for the papyrus mask of the fragment.

        Returns:
            str: The file name for the papyrus mask.
        """
        return "mask.png"

    def load_ink_labels_as_img(self) -> Image:
        """Load ink labels of the fragment as a PIL.Image.

        Returns:
            Image: The ink labels of the fragment.
        """
        return Image.open(self.ink_labels_path)

    def load_ink_labels(self) -> np.ndarray:
        """Load ink labels of the fragment as a NumPy array.

        Returns:
            np.ndarray: The ink labels of the fragment.
        """
        return np.array(self.load_ink_labels_as_img(), dtype=bool)

    def load_ir_img_as_img(self) -> Image:
        """Load the infrared (IR) image of the fragment as a PIL.Image.

        Returns:
            Image: The IR image of the fragment.
        """
        return Image.open(self.ir_img_path)

    def load_ir_img(self) -> np.ndarray:
        """Load the infrared (IR) image of the fragment as a NumPy array.

        Returns:
            np.ndarray: The IR image of the fragment.
        """
        return np.array(self.load_ir_img_as_img())

    @property
    def voxel_size_microns(self) -> float:
        """Get the voxel size in microns.

        Returns:
            float: The voxel size in microns.
        """
        return VOXEL_SIZE_MICRONS

    @property
    def voxel_size(self) -> pint.Quantity:
        """Get the voxel size as a pint quantity.

        Returns:
            pint.Quantity: The voxel size, using microns as the unit.
        """
        return VOXEL_SIZE

    @property
    def ppm_path(self) -> Path:
        return self.volume_dir_path / "result.ppm"

    @property
    def mesh_path(self) -> Path:
        return self.volume_dir_path / "result.obj"

    def __repr__(self):
        return f"{self.__class__.__name__}(fragment_id={self.fragment_id}, shape={self.shape})"
