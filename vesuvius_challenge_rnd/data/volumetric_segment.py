from typing import TYPE_CHECKING

import json
import os
from abc import ABC, abstractmethod
from functools import cached_property
from pathlib import Path

import numpy as np
import pint
from PIL import Image
from tifffile import TiffSequence, ZarrFileSequenceStore, imread

from vesuvius_challenge_rnd.data.ppm import PPM
from vesuvius_challenge_rnd.data.preprocessing import preprocess_subvolume
from vesuvius_challenge_rnd.data.util import create_tempfile_name

if TYPE_CHECKING:
    import trimesh


# Ignore PIL warnings about large images.
Image.MAX_IMAGE_PIXELS = 10_000_000_000


class VolumetricSegment(ABC):
    """Abstract base class of a volumetric segment."""

    surface_volume_dir_name = "surface_volume"  # The name of the surface volume directory name.

    def __init__(self, data_dir: Path, segment_name: str):
        """Initialize the volumetric segment.

        Args:
            data_dir (Path): The directory containing data.
            segment_name (str): The name of the segment.
        """
        if not data_dir.is_dir():
            raise NotADirectoryError(f"The data directory {data_dir} does not exist.")

        self.data_dir = data_dir
        self.segment_name = segment_name

        if not self.volume_dir_path.is_dir():
            NotADirectoryError(f"The volume data directory {self.volume_dir_path} does not exist.")
        if len(self.all_surface_volume_paths) == 0:
            raise ValueError(
                f"No surface volume paths found for segment {self.segment_name} in {self.volume_dir_path}"
            )

        self._tiff_sequence = TiffSequence(files=self.all_surface_volume_paths, mode="r")
        self._zarr = self.tiff_sequence.aszarr()

    @property
    @abstractmethod
    def papyrus_mask_file_name(self) -> str:
        """The papyrus mask file name directory name.

        Returns:
            str: The mask file name.
        """
        raise NotImplementedError("Child classes must implement this.")

    @property
    @abstractmethod
    def ppm_path(self) -> Path:
        """The PPM file path.

        Returns:
            Path: The PPM file path.
        """
        raise NotImplementedError("Child classes must implement this.")

    @property
    @abstractmethod
    def mesh_path(self) -> Path:
        """The PPM file path.

        Returns:
            Path: The mesh (.obj) file path.
        """
        raise NotImplementedError("Child classes must implement this.")

    @property
    def volume_dir_path(self) -> Path:
        """The volumetric segment data directory path.

        Returns:
            Path: Path to the volume directory.
        """
        return self.data_dir / self.segment_name

    @property
    def all_surface_volume_paths(self) -> list[Path]:
        """All surface volume paths.

        Returns:
            list[Path]: A list of paths to surface volumes.
        """
        surface_volume_dir = self.volume_dir_path / self.surface_volume_dir_name
        return list(sorted(surface_volume_dir.glob("*.tif"), key=lambda x: int(x.stem)))

    @property
    def papyrus_mask_path(self) -> Path:
        """The papyrus mask path.

        Returns:
            Path: Path to the papyrus mask.
        """
        return self.volume_dir_path / self.papyrus_mask_file_name

    def load_surface_vol_paths(
        self, z_start: int | None = None, z_end: int | None = None
    ) -> list[Path]:
        """Load surface volume paths.

        Args:
            z_start (int | None, optional): The start of the z-range. Defaults to None.
            z_end (int | None, optional): The end of the z-range. Defaults to None.

        Returns:
            list[Path]: A list of paths to the surface volumes.
        """
        return self.all_surface_volume_paths[z_start:z_end]

    def load_volume(
        self, z_start: int | None = None, z_end: int | None = None, preprocess: bool = True
    ) -> np.ndarray:
        """Load a volumetric segment as an array.

        Args:
            z_start (int | None, optional): The start of the z-range. Defaults to None.
            z_end (int | None, optional): The end of the z-range. Defaults to None.
            preprocess (bool, optional): Whether to apply sub-volume preprocessing. Defaults to True.

        Returns:
            np.ndarray: The volumetric segment array.
        """
        surface_volume_paths = self.load_surface_vol_paths(z_start=z_start, z_end=z_end)
        image_stack = imread(surface_volume_paths)
        if preprocess:
            image_stack = preprocess_subvolume(preprocess)
        return image_stack

    def load_volume_as_memmap(
        self, z_start: int | None = None, z_end: int | None = None
    ) -> np.memmap:
        """Load a volume (image stack) as a memory-mapped file.

        Args:
            z_start (int | None, optional): The start of the z-range. Defaults to None.
            z_end (int | None, optional): The end of the z-range. Defaults to None.
        """
        surface_volume_paths = self.load_surface_vol_paths(z_start=z_start, z_end=z_end)
        tiff_sequence = TiffSequence(files=surface_volume_paths, mode="r")

        tempdir = os.environ.get("MEMMAP_DIR")
        if tempdir is None:
            image_stack_ref = tiff_sequence.asarray(out="memmap")
        else:
            temp_file_name = create_tempfile_name(tempdir)
            image_stack_ref = tiff_sequence.asarray(
                out=f"{temp_file_name}_volume_{self.segment_name}.memmap"
            )
        assert isinstance(image_stack_ref, np.memmap)
        return image_stack_ref

    def load_mask_as_img(self) -> Image:
        """Load the mask as a PIL.Image.

        Returns:
            Image: The mask as an image.
        """
        return Image.open(self.papyrus_mask_path).convert("1")

    def load_mask(self) -> np.ndarray:
        """Load a fragment papyrus mask.

        Returns:
            np.ndarray: The mask array.
        """
        return np.array(self.load_mask_as_img(), dtype=bool)

    def load_ppm(self, dtype: np.dtype = np.float32) -> PPM:
        """Load the associated PPM object."""
        ppm = PPM.from_path(str(self.ppm_path))
        ppm.data = ppm.data.astype(dtype)
        return ppm

    def load_ppm_as_memmap(self) -> PPM:
        """Load the associated PPM object with the data as a memory-map."""
        ppm = PPM.from_path(str(self.ppm_path), lazy_load=True)
        ppm.load_ppm_data_as_memmap()
        return ppm

    def load_mesh(self) -> trimesh.Trimesh if TYPE_CHECKING else None:
        """Load the associated triangular mesh (obj) file."""
        import trimesh

        return trimesh.load_mesh(self.mesh_path)

    @property
    def tiff_sequence(self) -> TiffSequence:
        """The volumetric segment as a TIFF Sequence.

        Returns:
            TiffSequence: The sequence of TIFF files representing the volumetric segment.
        """
        return self._tiff_sequence

    @property
    def zarr(self) -> ZarrFileSequenceStore:
        """The volumetric segment as a Zarr.

        Returns:
            ZarrFileSequenceStore: The Zarr file sequence store representing the volumetric segment.
        """
        return self._zarr

    @cached_property
    def shape(self) -> tuple[int, int, int]:
        """The shape of the volumetric segment.

        Returns:
            tuple[int, int, int]: The shape of the volumetric segment in the form (z, y, x).
        """
        z_array = json.loads(self.zarr[".zarray"])
        return tuple(z_array["shape"])

    @cached_property
    def dtype(self) -> np.dtype:
        """The data type of the volumetric segment.

        Returns:
            np.dtype: The data type of the volumetric segment..
        """
        z_array = json.loads(self.zarr[".zarray"])
        return np.dtype(z_array["dtype"])

    @property
    def voxel_size_microns(self) -> float:
        """Voxel size in microns.

        Raises:
            NotImplementedError: This property must be implemented in a subclass.

        Returns:
            float: The voxel size in microns.
        """
        raise NotImplementedError

    @property
    def voxel_size(self) -> pint.Quantity:
        """Voxel size as a pint quantity.

        Raises:
            NotImplementedError: This property must be implemented in a subclass.

        Returns:
            pint.Quantity: The voxel size as a physical quantity.
        """
        raise NotImplementedError

    @property
    def n_slices(self) -> int:
        """The number of slices (layers) of the volumetric segment.

        Returns:
            int: The total number of slices in the segment.
        """
        return self.shape[0]

    @property
    def surface_shape(self) -> tuple[int, int]:
        """The shape of the surface of the volumetric segment.

        Returns:
            tuple[int, int]: The shape of the surface in the form (y, x).
        """
        return self.shape[-2:]

    @property
    def ndim(self) -> int:
        """The number of dimensions of the volumetric segment.

        Returns:
            int: The total number of dimensions, typically 3 for a volumetric segment.
        """
        return len(self.shape)

    def __repr__(self):
        return f"{self.__class__.__name__}(segment_name={self.segment_name}, shape={self.shape})"
