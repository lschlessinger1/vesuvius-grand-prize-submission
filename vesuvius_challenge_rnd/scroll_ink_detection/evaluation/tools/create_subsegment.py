import argparse
from pathlib import Path

import cv2
import numpy as np
from skimage.util import img_as_ubyte
from tifffile import tifffile

from vesuvius_challenge_rnd.data.scroll import ScrollSegment, create_scroll_segment


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


def _imwrite_uint8(array: np.ndarray, output_path: str | Path):
    _verify_is_dtype(array, dtype=np.uint8)
    cv2.imwrite(str(output_path), array)


def save_mask(mask: np.ndarray, output_path: Path) -> None:
    """Save the papyrus mask to the preprocessing directory."""
    _imwrite_uint8(mask, output_path)
    print(f"Saved mask of shape {mask.shape}.")


def save_surface_volumes(surface_volumes: np.ndarray, output_dir: Path) -> None:
    """Save the surface volumes to the preprocessing directory."""
    _verify_is_dtype(surface_volumes, dtype=np.uint16)
    output_dir.mkdir(exist_ok=True, parents=True)

    for i, img in enumerate(surface_volumes):
        output_path = output_dir / f"{str(i).zfill(2)}.tif"
        tifffile.imwrite(output_path, img)

    print(f"Saved {len(surface_volumes)} surface volumes.")


def _prepare_subvolume(
    segment: ScrollSegment, u1: int, u2: int, v1: int, v2: int, use_memmap: bool = True
) -> np.ndarray:
    if use_memmap:
        volume = segment.load_volume_as_memmap()
    else:
        volume = segment.load_volume(preprocess=False)
    return volume[:, u1:u2, v1:v2]


def _prepare_subsegment_mask(
    segment: ScrollSegment, u1: int, u2: int, v1: int, v2: int
) -> np.ndarray:
    mask = segment.load_mask()
    subsegment_mask = mask[u1:u2, v1:v2]
    return img_as_ubyte(subsegment_mask)


def create_subsegment(
    segment_name: str,
    scroll_id: str,
    u1: int,
    u2: int | None,
    v1: int,
    v2: int | None,
    column_id: str,
    skip_if_exists: bool = True,
    use_memmap: bool = True,
) -> None:
    print(f"Creating segment {segment_name}")
    segment = create_scroll_segment(scroll_id=scroll_id, segment_name=segment_name)

    output_segment_name = f"{segment_name}_{column_id}"
    output_dir = segment.volume_dir_path.with_name(output_segment_name)
    if skip_if_exists and output_dir.exists() and any(output_dir.iterdir()):
        print(f"Skipping existing sub-segment: {output_segment_name}")
        return

    if u2 is None:
        u2 = segment.surface_shape[0]

    if v2 is None:
        v2 = segment.surface_shape[1]

    print(f"Creating subvolume and mask u: {u1}-{u2}; v: {v1}-{v2}")
    subvolume = _prepare_subvolume(segment, u1, u2, v1, v2, use_memmap=use_memmap)
    subsegment_mask = _prepare_subsegment_mask(segment, u1, u2, v1, v2)

    output_dir.mkdir(exist_ok=True, parents=True)

    output_mask_name = segment.papyrus_mask_file_name.replace(
        segment.segment_name, output_segment_name
    )
    output_mask_path = output_dir / output_mask_name
    print(f"Saving segment mask to {output_mask_path}")
    save_mask(subsegment_mask, output_mask_path)

    output_surface_volume_path = output_dir / segment.surface_volume_dir_name
    print(f"Saving surface volume to {output_surface_volume_path}")
    save_surface_volumes(subvolume, output_surface_volume_path)


def main():
    parser = _set_up_parser()
    args = parser.parse_args()
    create_subsegment(
        args.segment_id,
        args.scroll_id,
        args.u1,
        args.u2,
        args.v1,
        args.v2,
        args.column_id,
        skip_if_exists=args.skip_if_exists,
        use_memmap=not args.load_in_memory,
    )


def _set_up_parser() -> argparse.ArgumentParser:
    """Set up argument parser for command-line interface.

    Returns:
        ArgumentParser: Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description="Generate a subsegment for a given scroll segment."
    )
    parser.add_argument(
        "u1",
        type=int,
        help="The starting index (vertical axis) in the UV-coordinate system to cut the sub-segment.",
    )
    parser.add_argument(
        "u2",
        type=int,
        help="The ending index (vertical axis) in the UV-coordinate system to cut the sub-segment.",
    )
    parser.add_argument(
        "v1",
        type=int,
        help="The starting index (horizontal axis) in the UV-coordinate system to cut the sub-segment.",
    )
    parser.add_argument(
        "v2",
        type=int,
        help="The starting index (horizontal axis) in the UV-coordinate system to cut the sub-segment.",
    )
    parser.add_argument("segment_id", type=str, help="The segment ID to cut.")
    parser.add_argument(
        "--scroll_id", type=str, default="1", help="The scroll ID for which the segment belongs"
    )
    parser.add_argument(
        "--column_id",
        type=str,
        default="C0",
        help="The column ID for the new segment. The new segment name will be <segment_id>_<column_id>",
    )
    parser.add_argument(
        "--skip_if_exists",
        action="store_true",
        help="Skip the creation of the sub-segment if it already exists.",
    )
    parser.add_argument(
        "--load_in_memory",
        action="store_true",
        help="Load the image stack in RAM instead of as a memory-mapped file.",
    )
    return parser


if __name__ == "__main__":
    main()
