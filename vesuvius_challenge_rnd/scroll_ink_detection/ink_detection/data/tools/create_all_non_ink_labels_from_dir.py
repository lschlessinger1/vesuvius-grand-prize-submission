"""Create all non-ink labels from an ink label directory."""
import argparse
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

from vesuvius_challenge_rnd.scroll_ink_detection.ink_detection.data.tools.create_non_ink_labels import (
    DEFAULT_OUTPUT_DIR,
    create_non_ink_labels,
)


def create_all_non_ink_labels_from_dir(
    ink_label_dir: Path, output_dir: Path = DEFAULT_OUTPUT_DIR, dilation_kernel_size: int = 256
) -> None:
    """Create all non-ink labels from an ink label directory."""
    pattern_1 = "*_inklabels.png"
    pattern_2 = "*_inklabels_0.png"
    ink_labels_paths = [
        file for pattern in [pattern_1, pattern_2] for file in ink_label_dir.glob(pattern)
    ]
    print(f"Found {len(ink_labels_paths)} ink label paths in {ink_label_dir}.")
    output_dir.mkdir(exist_ok=True, parents=True)
    for path in tqdm(ink_labels_paths, desc="Generating non-ink labels"):
        create_non_ink_labels(
            path,
            output_dir=output_dir,
            dilation_kernel_size=dilation_kernel_size,
        )


def main():
    load_dotenv()
    parser = _set_up_parser()
    args = parser.parse_args()
    create_all_non_ink_labels_from_dir(
        Path(args.ink_label_dir),
        Path(args.output_dir),
        dilation_kernel_size=args.dilation_kernel_size,
    )


def _set_up_parser() -> argparse.ArgumentParser:
    """Set up argument parser for command-line interface.

    Returns:
        ArgumentParser: Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description="Generate non-ink masks for many scroll segments from an ink label directory."
    )
    parser.add_argument(
        "ink_label_dir",
        type=str,
        help="The ink labels directory.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="The output non-ink labels file path.",
    )
    parser.add_argument(
        "-k",
        "--dilation_kernel_size",
        type=int,
        default=256,
        help="The non-ink dilation kernel size.",
    )
    return parser


if __name__ == "__main__":
    main()
