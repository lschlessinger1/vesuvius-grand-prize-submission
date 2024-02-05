"""Per-pixel map adapted from https://github.com/educelab/ink-id/blob/53e1d696d9270cc13e3c3674939f5b60eb78faaa/inkid/data/ppm.py"""
# For PPM.initialized_ppms https://stackoverflow.com/a/33533514
from __future__ import annotations

from typing import ClassVar

import logging
import re
import struct
from io import BytesIO
from pathlib import Path

import numpy as np
from tqdm import tqdm

from vesuvius_challenge_rnd.data.util import find_pattern_offset, get_raw_data_from_file_or_url


class PPM:
    """Class to handle PPM (Per-pixel map) data."""

    initialized_ppms: ClassVar[dict[str, PPM]] = dict()

    def __init__(self, path: str, lazy_load: bool = False):
        """Initialize a PPM object.

        Args:
            path (str): Path to the PPM file.
            lazy_load (bool, optional): Whether to load data lazily. Defaults to False.
        """
        self._path = path

        header = PPM.parse_ppm_header(path)
        self.width: int = header["width"]
        self.height: int = header["height"]
        self.dim: int = header["dim"]
        self.ordered: bool = header["ordered"]
        self.type: str = header["type"]
        self.version: str = header["version"]

        self.data: np.typing.ArrayLike | None = None

        logging.info(
            f"Initialized PPM for {self._path} with width {self.width}, "
            f"height {self.height}, dim {self.dim}"
        )

        if not lazy_load:
            self.ensure_loaded()

    def is_loaded(self) -> bool:
        """Check if the PPM data is loaded.

        Returns:
            bool: True if data is loaded, otherwise False.
        """
        return self.data is not None

    def ensure_loaded(self) -> None:
        """Ensure that the PPM data is loaded."""
        if not self.is_loaded():
            self.load_ppm_data()

    @classmethod
    def from_path(cls, path: str, lazy_load: bool = False) -> PPM:
        """Create a PPM object from a path, with optional lazy loading.

        Args:
            path (str): Path to the PPM file.
            lazy_load (bool, optional): Whether to load data lazily. Defaults to False.

        Returns:
            PPM: The PPM object.
        """
        if path in cls.initialized_ppms:
            return cls.initialized_ppms[path]
        cls.initialized_ppms[path] = PPM(path, lazy_load=lazy_load)
        return cls.initialized_ppms[path]

    @staticmethod
    def parse_ppm_header(filename: str) -> dict:
        """Parse the header of a PPM file.

        Args:
            filename (str): Path to the PPM file.

        Returns:
            dict: The parsed header information.
        """
        comments_re = re.compile("^#")
        width_re = re.compile("^width")
        height_re = re.compile("^height")
        dim_re = re.compile("^dim")
        ordered_re = re.compile("^ordered")
        type_re = re.compile("^type")
        version_re = re.compile("^version")
        header_terminator_re = re.compile("^<>$")

        width, height, dim, ordered, val_type, version = [None] * 6

        data = get_raw_data_from_file_or_url(filename)
        while True:
            line = data.readline().decode("utf-8")
            if comments_re.match(line):
                pass
            elif width_re.match(line):
                width = int(line.split(": ")[1])
            elif height_re.match(line):
                height = int(line.split(": ")[1])
            elif dim_re.match(line):
                dim = int(line.split(": ")[1])
            elif ordered_re.match(line):
                ordered = line.split(": ")[1].strip() == "true"
            elif type_re.match(line):
                val_type = line.split(": ")[1].strip()
                assert val_type in ["double"]
            elif version_re.match(line):
                version = line.split(": ")[1].strip()
            elif header_terminator_re.match(line):
                break
            else:
                logging.warning(f"PPM header contains unknown line: {line.strip()}")

        return {
            "width": width,
            "height": height,
            "dim": dim,
            "ordered": ordered,
            "type": val_type,
            "version": version,
        }

    @staticmethod
    def write_ppm_from_data(
        path: str,
        data: np.typing.ArrayLike,
        width: int,
        height: int,
        dim: int,
        ordered: bool = True,
        version: str = "1.0",
        pbar: bool = True,
    ) -> None:
        """Write PPM data to a file.

        Args:
            path (str): The file path to write the PPM data.
            data (np.typing.ArrayLike): The data array.
            width (int): The width of the PPM.
            height (int): The height of the PPM.
            dim (int): The dimension of the PPM.
            ordered (bool, optional): Whether the data is ordered. Defaults to True.
            version (str, optional): The PPM version. Defaults to "1.0".
            pbar (bool, optional): Whether to show a progress bar. Defaults to True.
        """
        with open(path, "wb") as f:
            logging.info(f"Writing PPM to file {path}...")
            f.write(f"width: {width}\n".encode())
            f.write(f"height: {height}\n".encode())
            f.write(f"dim: {dim}\n".encode())
            f.write("ordered: {}\n".format("true" if ordered else "false").encode("utf-8"))
            f.write(b"type: double\n")
            f.write(f"version: {version}\n".encode())
            f.write(b"<>\n")
            y_iter = range(height)
            if pbar:
                y_iter = tqdm(y_iter, desc="Writing PPM...")
            for y in y_iter:
                for x in range(width):
                    for idx in range(dim):
                        f.write(struct.pack("d", data[y, x, idx]))

    def load_ppm_data(self, count: int = -1) -> None:
        """Read the PPM file data and store it in the PPM object.

        The data is stored in an internal array indexed by [y, x, idx]
        where idx is an index into an array of size dim.

        Example: For a PPM of dimension 6 to store 3D points and
        normals, the first component of the normal vector for the PPM
        origin would be at self._data[0, 0, 3].

        Parameters:
            count : int, optional
                Number of items to read. ``-1`` means all data in the buffer.

        Raises:
            ValueError: If the header terminator "<>\n" is not found in the data stream.
            IOError: If unable to read the file or URL specified by ppm_path.

        Notes:
        - Assumes that the PPM data has a header that ends with "<>\n".
        - Assumes that the pixel data is stored as float64.
        - This function relies on `get_raw_data_from_file_or_url` to get a BytesIO object for the data,
          and `find_ppm_header_terminator_offset` to find the header terminator offset.
        """
        logging.info(
            f"Loading PPM data for {self._path} with width {self.width}, "
            f"height {self.height}, dim {self.dim}..."
        )

        data_io = get_raw_data_from_file_or_url(self._path)
        n_offset_bytes = _find_ppm_header_terminator_offset(data_io)
        self.data = np.frombuffer(
            data_io.getbuffer(), dtype=np.float64, count=count, offset=n_offset_bytes
        ).reshape((self.height, self.width, self.dim))

    def load_ppm_data_as_memmap(self) -> None:
        """Read the PPM file data and store it in the PPM object as a memory map."""
        data_io = get_raw_data_from_file_or_url(self._path)
        n_offset_bytes = _find_ppm_header_terminator_offset(data_io)
        self.data = np.memmap(
            self._path,
            dtype=np.float64,
            mode="r",
            offset=n_offset_bytes,
            shape=(self.height, self.width, self.dim),
        )

    def get_point_with_normal(self, ppm_x: int, ppm_y: int) -> np.typing.ArrayLike:
        """Get the point along with its normal at given coordinates.

        Args:
            ppm_x (int): The x-coordinate.
            ppm_y (int): The y-coordinate.

        Returns:
            np.typing.ArrayLike: The point with its normal.
        """
        self.ensure_loaded()
        return self.data[ppm_y][ppm_x]

    def get_points(self) -> np.typing.ArrayLike:
        """Get all 3D points in the PPM.

        Returns:
            np.typing.ArrayLike: The 3D points.
        """
        self.ensure_loaded()
        return self.data[:, :, :3]

    def get_surface_normals(self) -> np.typing.ArrayLike:
        """Get all surface normals in the PPM.

        Returns:
            np.typing.ArrayLike: The surface normals.
        """
        self.ensure_loaded()
        return self.data[:, :, 3:]

    def scale_down_by(self, scale_factor: float, pbar: bool = True) -> None:
        """Scale down the PPM by a given factor.

        Args:
            scale_factor (float): The scale factor.
            pbar (bool, optional): Whether to show a progress bar. Defaults to True.
        """
        self.ensure_loaded()

        self.width //= scale_factor
        self.height //= scale_factor

        new_data = np.empty((self.height, self.width, self.dim))

        logging.info(f"Downscaling PPM by factor of {scale_factor} on all axes...")
        y_iter = range(self.height)
        if pbar:
            y_iter = tqdm(y_iter, desc="Downscaling PPM...")
        for y in y_iter:
            for x in range(self.width):
                for idx in range(self.dim):
                    new_data[y, x, idx] = self.data[y * scale_factor, x * scale_factor, idx]

        self.data = new_data

    def translate(self, dx: int, dy: int, dz: int, pbar: bool = True) -> None:
        """Translate the PPM by given distances along each axis.

        Args:
            dx (int): Distance to translate along the x-axis.
            dy (int): Distance to translate along the y-axis.
            dz (int): Distance to translate along the z-axis.
            pbar (bool, optional): Whether to show a progress bar. Defaults to True.
        """
        y_iter = range(self.height)
        if pbar:
            y_iter = tqdm(y_iter, desc="Translating PPM...")
        for ppm_y in y_iter:
            for ppm_x in range(self.width):
                if np.any(self.data[ppm_y, ppm_x]):  # Leave empty pixels unchanged
                    vol_x, vol_y, vol_z = self.data[ppm_y, ppm_x, 0:3]
                    self.data[ppm_y, ppm_x, 0] = vol_x + dx
                    self.data[ppm_y, ppm_x, 1] = vol_y + dy
                    self.data[ppm_y, ppm_x, 2] = vol_z + dz

    def write(self, filename: Path | str, pbar: bool = True) -> None:
        """Write the PPM object to a file.

        Args:
            filename (Union[Path, str]): The file path or name.
            pbar (bool, optional): Whether to show a progress bar. Defaults to True.
        """
        self.ensure_loaded()

        with open(filename, "wb") as f:
            logging.info(f"Writing PPM to file {filename}...")
            f.write(f"width: {self.width}\n".encode())
            f.write(f"height: {self.height}\n".encode())
            f.write(f"dim: {self.dim}\n".encode())
            f.write("ordered: {}\n".format("true" if self.ordered else "false").encode("utf-8"))
            f.write(b"type: double\n")
            f.write(f"version: {self.version}\n".encode())
            f.write(b"<>\n")
            y_iter = range(self.height)
            if pbar:
                y_iter = tqdm(y_iter, desc="Saving PPM...")
            for y in y_iter:
                for x in range(self.width):
                    for idx in range(self.dim):
                        f.write(struct.pack("d", self.data[y, x, idx]))

    def __getitem__(self, coords: tuple[int, int, int]) -> float:
        self.ensure_loaded()
        return self.data[coords]

    def __setitem__(self, coords: tuple[int, int, int], value: float) -> None:
        self.ensure_loaded()
        self.data[coords] = value

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PPM):
            return False
        self.ensure_loaded()
        other.ensure_loaded()
        return np.array_equal(self.data, other.data)

    def __len__(self) -> int:
        """The number of rows."""
        return self.width * self.height

    def __contains__(self, value: float) -> bool:
        self.ensure_loaded()
        return value in self.data

    def __repr__(self) -> str:
        loaded_status = "Loaded" if self.is_loaded() else "Not Loaded"
        return (
            f"PPM(path={self._path!r}, width={self.width}, height={self.height}, "
            f"dim={self.dim}, ordered={self.ordered}, type={self.type!r}, "
            f"version={self.version!r}, status={loaded_status})"
        )


def _find_ppm_header_terminator_offset(data_stream: BytesIO) -> int:
    """
    Find the position of the header terminator "<>\n" in the data.

    Parameters:
    data_stream (BytesIO): The BytesIO object containing the data stream.

    Returns:
    int: The inclusive offset of the first occurrence of the header terminator "<>\n".

    Raises:
    ValueError: If the header terminator "<>\n" is not found in the data stream.
    """
    pattern_to_search = b"<>\n"
    return find_pattern_offset(data_stream, pattern_to_search)
