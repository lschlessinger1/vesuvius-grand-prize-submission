from typing import Literal

import gc
import os
from pathlib import Path

import numpy as np

from vesuvius_challenge_rnd import DATA_DIR
from vesuvius_challenge_rnd.data import ScrollSegment

MEMMAP_DIR = DATA_DIR / "memmaps"


def segment_to_memmap_path(
    segment: ScrollSegment,
    memmap_dir: Path = MEMMAP_DIR,
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
    return memmap_dir / segment.scroll_id / f"{stem}.npy"


def save_segment_as_memmap(
    segment: ScrollSegment,
    output_memmap_path: Path | None,
    load_in_memory: bool = False,
    z_min: int | None = None,
    z_max: int | None = None,
    memmap_dir: Path = MEMMAP_DIR,
) -> np.memmap:
    if output_memmap_path is None:
        output_memmap_path = segment_to_memmap_path(
            segment, memmap_dir=memmap_dir, z_min=z_min, z_max=z_max
        )

    if load_in_memory:
        img_stack = segment.load_volume(z_start=z_min, z_end=z_max, preprocess=False)
    else:
        img_stack = segment.load_volume_as_memmap(z_start=z_min, z_end=z_max)

    output_memmap_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(output_memmap_path), img_stack)

    if not load_in_memory:
        delete_memmap(img_stack)

    return img_stack


def load_segment_as_memmap(
    memmap_path: Path,
    mmap_mode: Literal[None, "r+", "r", "w+", "c"] = "r",
) -> np.memmap:
    return np.load(str(memmap_path), mmap_mode=mmap_mode, allow_pickle=True)


def delete_memmap(img_stack_ref: np.memmap) -> None:
    filename = img_stack_ref.filename
    img_stack_ref._mmap.close()
    del img_stack_ref
    gc.collect()
    if os.path.exists(filename):
        os.remove(filename)
