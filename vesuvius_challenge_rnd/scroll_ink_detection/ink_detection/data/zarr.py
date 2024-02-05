import logging
from pathlib import Path

import zarr

from vesuvius_challenge_rnd import DATA_DIR
from vesuvius_challenge_rnd.data.scroll import ScrollSegment
from vesuvius_challenge_rnd.scroll_ink_detection.ink_detection.data.memmap import delete_memmap

ZARR_ARRAY_DIR = DATA_DIR / "zarrs"


def segment_to_zarr_path(
    segment: ScrollSegment,
    zarr_dir: Path = ZARR_ARRAY_DIR,
    z_min: int | None = None,
    z_max: int | None = None,
) -> Path:
    if (z_max is None and z_min is not None) or (z_max is not None and z_min is None):
        raise ValueError("z_max and z_min must be both None or be set to integer values.")

    stem = segment.segment_name if not segment.is_subsegment else segment.segment_name_orig
    if z_max is not None and z_min is not None:
        if z_min > z_max:
            raise ValueError(f"z_min ({z_min}) must be less than z_max ({z_max}).")
        stem = "_".join((stem, f"{z_min}-{z_max}"))
    return zarr_dir / segment.scroll_id / f"{stem}.zarr"


def save_segment_as_zarr_array(
    segment: ScrollSegment,
    output_zarr_path: Path | None,
    z_chunk_size: int = 4,
    y_chunk_size: int = 512,
    x_chunk_size: int = 512,
    load_in_memory: bool = False,
    z_min: int | None = None,
    z_max: int | None = None,
    zarr_dir: Path = ZARR_ARRAY_DIR,
) -> zarr.Array:
    if output_zarr_path is None:
        output_zarr_path = segment_to_zarr_path(
            segment, z_min=z_min, z_max=z_max, zarr_dir=zarr_dir
        )

    if load_in_memory:
        img_stack = segment.load_volume(z_start=z_min, z_end=z_max, preprocess=False)
    else:
        img_stack = segment.load_volume_as_memmap(z_start=z_min, z_end=z_max)

    zarr_array = zarr.open(
        str(output_zarr_path),
        mode="w",
        shape=img_stack.shape,
        dtype=img_stack.dtype,
        chunks=(z_chunk_size, y_chunk_size, x_chunk_size),
    )
    zarr_array[:] = img_stack

    if not load_in_memory:
        delete_memmap(img_stack)

    return zarr_array


def load_segment_as_zarr_array(
    zarr_path: Path, chunks: tuple[int, int, int] | int | bool = True
) -> zarr.Array:
    return zarr.open(str(zarr_path), mode="r", chunks=chunks)


def create_or_load_img_stack_zarr(
    segment: ScrollSegment,
    zarr_dir: Path = ZARR_ARRAY_DIR,
    chunks_load: tuple[int, int, int] | int | bool = True,
    z_chunk_save_size: int = 4,
    y_chunk_save_size: int = 512,
    x_chunk_save_size: int = 512,
) -> zarr.Array:
    zarr_path = segment_to_zarr_path(segment, zarr_dir=zarr_dir)
    if zarr_path.exists():
        logging.info(f"Found existing zarr for segment {segment.segment_name}.")
        return load_segment_as_zarr_array(zarr_path, chunks=chunks_load)
    else:
        logging.info(
            f"Did not find existing zarr for segment {segment.segment_name}. A new zarr array will be created."
        )
        return save_segment_as_zarr_array(
            segment,
            output_zarr_path=zarr_path,
            z_chunk_size=z_chunk_save_size,
            y_chunk_size=y_chunk_save_size,
            x_chunk_size=x_chunk_save_size,
        )
