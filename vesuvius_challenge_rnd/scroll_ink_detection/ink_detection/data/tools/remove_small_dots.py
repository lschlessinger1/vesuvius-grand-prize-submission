"""Merge two sets of binary labels into a single mask."""
import argparse
from pathlib import Path

import cv2
from skimage.util import img_as_ubyte

from vesuvius_challenge_rnd.data.preprocessors.fragment_preprocessor import _imwrite_uint8
from vesuvius_challenge_rnd.scroll_ink_detection.ink_detection.data.tools.label_utils import (
    remove_small_components,
)

DEFAULT_OUTPUT_PATH = Path("processed_mask.png")


def remove_small_dots(
    mask_path: Path, min_size: int = 100, output_path: Path = DEFAULT_OUTPUT_PATH
) -> None:
    # Load masks.
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    # Remove small dots.
    output_mask = img_as_ubyte(remove_small_components(mask, min_size=min_size))

    # Save masks.
    print(f"Saving processed mask to {output_path.resolve()}")
    _imwrite_uint8(output_mask, output_path)
    print(f"Saved mask of shape {output_mask.shape}.")


def main():
    parser = _set_up_parser()
    args = parser.parse_args()
    remove_small_dots(
        Path(args.mask_path),
        args.min_size,
        output_path=Path(args.output_path),
    )


def _set_up_parser() -> argparse.ArgumentParser:
    """Set up argument parser for command-line interface.

    Returns:
        ArgumentParser: Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description="Remove small dots from a binary mask and save it as a uint8."
    )
    parser.add_argument(
        "mask_path",
        type=str,
        help="The mask path.",
    )
    parser.add_argument(
        "-m",
        "--min_size",
        type=int,
        default=1000,
        help="The minimum connected component size.",
    )
    parser.add_argument(
        "-o",
        "--output_path",
        type=str,
        default=DEFAULT_OUTPUT_PATH,
        help="The path to the processed mask output file.",
    )
    return parser


if __name__ == "__main__":
    main()
