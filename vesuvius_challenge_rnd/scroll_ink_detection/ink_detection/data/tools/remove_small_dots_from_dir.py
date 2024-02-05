"""Merge two sets of binary labels into a single mask."""
import argparse
from pathlib import Path

import cv2
from skimage.util import img_as_ubyte
from tqdm import tqdm

from vesuvius_challenge_rnd.data.preprocessors.fragment_preprocessor import _imwrite_uint8
from vesuvius_challenge_rnd.scroll_ink_detection.ink_detection.data.tools.label_utils import (
    remove_small_components,
)
from vesuvius_challenge_rnd.scroll_ink_detection.ink_detection.data.tools.remove_small_dots import (
    remove_small_dots,
)

DEFAULT_OUTPUT_DIR = Path("masks_no_dots_removed")


def remove_small_from_dir(
    mask_dir: Path, min_size: int = 1000, output_dir: Path = DEFAULT_OUTPUT_DIR
) -> None:
    mask_paths = list(mask_dir.glob("*.png"))
    print(f"Found {len(mask_paths)} mask paths in {mask_dir}.")
    output_dir.mkdir(exist_ok=True, parents=True)
    for path in tqdm(mask_paths, desc="Removing small dots from masks..."):
        output_path = output_dir / path.name
        remove_small_dots(path, min_size=min_size, output_path=output_path)


def main():
    parser = _set_up_parser()
    args = parser.parse_args()
    remove_small_from_dir(
        Path(args.mask_dir),
        args.min_size,
        output_dir=Path(args.output_dir),
    )


def _set_up_parser() -> argparse.ArgumentParser:
    """Set up argument parser for command-line interface.

    Returns:
        ArgumentParser: Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description="Remove small dots from a directory of binary masks and save them."
    )
    parser.add_argument(
        "mask_dir",
        type=str,
        help="The mask directory.",
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
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="The path to the directory containing the processed mask output files.",
    )
    return parser


if __name__ == "__main__":
    main()
