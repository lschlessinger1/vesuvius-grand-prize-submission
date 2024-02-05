"""Merge two sets of binary labels into a single mask."""
import argparse
from pathlib import Path

import cv2
from skimage.util import img_as_ubyte

from vesuvius_challenge_rnd.data.preprocessors.fragment_preprocessor import _imwrite_uint8

DEFAULT_OUTPUT_PATH = Path("merged_mask.png")


def merge_masks(
    mask_path_1: Path, mask_path_2: Path, output_path: Path = DEFAULT_OUTPUT_PATH
) -> None:
    # Load masks.
    mask_1 = cv2.imread(str(mask_path_1), cv2.IMREAD_GRAYSCALE).astype(bool)
    mask_2 = cv2.imread(str(mask_path_2), cv2.IMREAD_GRAYSCALE).astype(bool)

    # Merge masks.
    output_mask = img_as_ubyte(mask_1 | mask_2)

    # Save masks.
    print(f"Saving merged mask to {output_path.resolve()}")
    _imwrite_uint8(output_mask, output_path)
    print(f"Saved mask of shape {output_mask.shape}.")


def main():
    parser = _set_up_parser()
    args = parser.parse_args()
    merge_masks(
        Path(args.mask_path_1),
        Path(args.mask_path_2),
        output_path=Path(args.output_path),
    )


def _set_up_parser() -> argparse.ArgumentParser:
    """Set up argument parser for command-line interface.

    Returns:
        ArgumentParser: Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description="Generate the union of two binary masks and save it as a uint8."
    )
    parser.add_argument(
        "mask_path_1",
        type=str,
        help="The first mask path.",
    )
    parser.add_argument(
        "mask_path_2",
        type=str,
        help="The second mask path.",
    )
    parser.add_argument(
        "-o",
        "--output_path",
        type=str,
        default=DEFAULT_OUTPUT_PATH,
        help="The path to the merged mask output file.",
    )
    return parser


if __name__ == "__main__":
    main()
