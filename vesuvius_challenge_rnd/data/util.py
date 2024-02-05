from typing import Any, TypeVar

import os
import re
import tempfile
from collections.abc import Sequence
from io import BytesIO
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
import requests
from PIL import Image

_T = TypeVar("_T")
_SequenceLike = Sequence[_T] | np.ndarray
_ScalarOrSequence = _T | _SequenceLike[_T]
_BoolLike_co = bool | np.bool_
_IntLike_co = _BoolLike_co | int | np.integer[Any]
_FloatLike_co = _IntLike_co | float | np.floating[Any]


def microns_to_indices(
    microns: _ScalarOrSequence[_FloatLike_co], voxel_size_microns: float
) -> np.integer[Any] | np.ndarray:
    """Convert spatial coordinates (in microns) to volumetric indices.

    This function takes spatial coordinates (in microns) and a given voxel size (also in microns)
    to convert these coordinates into integer indices representing their position within a
    volumetric space. This can be useful for converting real-world measurements into indices
    that can be used to index into a volumetric data structure like a 3D array.

    Args:
        microns: A scalar or sequence-like object of floating-point values representing spatial
                 coordinates in microns.
        voxel_size_microns: The size of a voxel in microns. It defines the conversion factor
                            between spatial coordinates and indices.

    Returns:
        np.integer[Any] | np.ndarray: Integer or array of integers representing the converted
                                      indices corresponding to the spatial coordinates.
    """
    return (np.asanyarray(microns) / voxel_size_microns).astype(int)


def indices_to_microns(
    indices: _ScalarOrSequence[_IntLike_co], voxel_size_microns: float
) -> np.floating[Any] | np.ndarray:
    """Convert volumetric indices to space (in microns).

    This function takes volumetric indices and a given voxel size (in microns) to convert
    these indices into real-world spatial coordinates. This can be useful for mapping from
    a volumetric data structure like a 3D array back to real-world measurements.

    Args:
        indices: A scalar or sequence-like object of integer values representing volumetric
                 indices within a 3D space.
        voxel_size_microns: The size of a voxel in microns. It defines the conversion factor
                            between indices and spatial coordinates.

    Returns:
        np.floating[Any] | np.ndarray: Float or array of floats representing the converted
                                       spatial coordinates corresponding to the volumetric indices.
    """
    return (np.asanyarray(indices) * voxel_size_microns).astype(float)


def get_raw_data_from_file_or_url(
    filename: str, return_relative_url: bool = False
) -> BytesIO | tuple[BytesIO, tuple]:
    """Return the raw file contents from a filename or URL.

    Supports absolute and relative file paths as well as the http and https
    protocols.

    """
    url = urlsplit(filename)
    is_windows_path = len(filename) > 1 and filename[1] == ":"
    if url.scheme in ("http", "https"):
        response = requests.get(filename)
        if response.status_code != 200:
            raise ValueError(f"Unable to fetch URL " f"(code={response.status_code}): {filename}")
        data = response.content
    elif url.scheme == "" or is_windows_path:
        with open(filename, "rb") as f:
            data = f.read()
    else:
        raise ValueError(f"Unsupported URL: {filename}")
    relative_url = (
        url.scheme,
        url.netloc,
        os.path.dirname(url.path),
        url.query,
        url.fragment,
    )
    if return_relative_url:
        return BytesIO(data), relative_url
    else:
        return BytesIO(data)


def find_pattern_offset(data_stream: BytesIO, pattern: bytes) -> int:
    """
    Find the inclusive offset of the first occurrence of a pattern in a BytesIO object.

    Parameters:
    data (BytesIO): The BytesIO object containing the data.
    pattern (bytes): The byte pattern to search for.

    Returns:
    int: The inclusive offset of the first occurrence of the pattern.

    Raises:
    ValueError: If the pattern is not found in the data.
    """
    compiled_pattern = re.compile(pattern)
    offset = 0

    while True:
        line = data_stream.readline()
        if not line:
            raise ValueError(
                f"The specified pattern {pattern.decode('utf-8')} was not found in the data."
            )

        match_result = compiled_pattern.search(line)
        if match_result:
            offset += match_result.start()
            return offset + (match_result.end() - match_result.start())

        offset += len(line)


def create_tempfile_name(tempdir: Path | str) -> str:
    # Ensure the directory exists
    if not os.path.exists(tempdir):
        os.makedirs(tempdir)

    # Generate a unique filename in the directory
    with tempfile.NamedTemporaryFile(dir=tempdir, delete=True) as tf:
        temp_filename = tf.name

    return temp_filename


def get_img_width_height(img_path: str | Path) -> tuple[int, int]:
    """
    Get the width and height of an image using PIL's Image.open method.

    Args:
        img_path (str | Path): The file path to the image.

    Returns:
        tuple[int, int]: A tuple containing the width and height of the image.

    Example:
        >>> get_img_width_height("path/to/image.png")
        (800, 600)
    """
    with Image.open(img_path) as img:
        return img.size
