import argparse
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv
from skimage.util import img_as_ubyte

from vesuvius_challenge_rnd.data.preprocessors.fragment_preprocessor import _imwrite_uint8
from vesuvius_challenge_rnd.scroll_ink_detection.ink_detection.data.ink_label_correction import (
    apply_model_based_label_correction,
)
from vesuvius_challenge_rnd.scroll_ink_detection.ink_detection.data.tools.label_utils import (
    create_non_ink_mask_from_ink_mask,
    load_ink_mask,
)

DEFAULT_OUTPUT_DIR = Path(__file__).parent


def create_non_ink_labels(
    ink_labels_path: Path,
    ink_pred_path: Path | None = None,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    ink_thresh: float = 0.05,
    dilation_kernel_size: int = 256,
    model_based_non_ink_correction_thresh: float = 0.99,
) -> None:
    print(f"Creating non-ink labels from {ink_labels_path}")
    ink_mask = (load_ink_mask(ink_labels_path) > ink_thresh).astype(np.uint8)
    non_ink_mask = create_non_ink_mask_from_ink_mask(
        ink_mask, dilation_kernel_size=dilation_kernel_size
    ).astype(bool)

    # Possibly auto-clean non-ink mask.
    if ink_pred_path is not None:
        ink_pred_path = Path(ink_pred_path)
        ink_pred = cv2.imread(str(ink_pred_path), cv2.IMREAD_GRAYSCALE)
        if ink_pred.shape != ink_mask.shape:
            raise ValueError(
                f"Ink predictions must have the same shape as the ink mask. Ink "
                f"preds has shape {ink_pred.shape} and ink labels shape {ink_mask.shape}."
            )
        print(f"Applying model-based non-ink label correction using {ink_pred_path}")
        _, non_ink_mask = apply_model_based_label_correction(
            ink_mask,
            non_ink_mask,
            ink_pred,
            ink_label_thresh=ink_thresh,
            model_based_non_ink_correction_thresh=model_based_non_ink_correction_thresh,
            clean_up_ink_labels=False,
            clean_up_non_ink_labels=True,
        )
    else:
        print(f"Skipping model-based label correction.")

    # Save non-ink mask.
    segment_name = ink_labels_path.stem.split("_inklabels")[0]
    output_path = output_dir / f"{segment_name}_papyrusnoninklabels.png"

    print(f"Saving non-ink mask to {output_path}")
    save_mask(img_as_ubyte(non_ink_mask), output_path)


def save_mask(mask: np.ndarray, output_path: Path) -> None:
    """Save the non-ink mask to the preprocessing directory."""
    _imwrite_uint8(mask, output_path)
    print(f"Saved non-ink mask of shape {mask.shape}.")


def main():
    load_dotenv()
    parser = _set_up_parser()
    args = parser.parse_args()
    create_non_ink_labels(
        Path(args.ink_labels_path),
        args.ink_pred_path,
        output_dir=Path(args.output_dir),
        dilation_kernel_size=args.dilation_kernel_size,
        model_based_non_ink_correction_thresh=args.model_based_non_ink_correction_thresh,
    )


def _set_up_parser() -> argparse.ArgumentParser:
    """Set up argument parser for command-line interface.

    Returns:
        ArgumentParser: Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description="Generate a non-ink mask for a scroll segment from an ink mask."
    )
    parser.add_argument(
        "ink_labels_path",
        type=str,
        help="The segment ink labels file path.",
    )
    parser.add_argument(
        "-p",
        "--ink_pred_path",
        type=str,
        help="The segment prediction file path.",
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
    parser.add_argument(
        "-t",
        "--model_based_non_ink_correction_thresh",
        type=float,
        default=0.99,
        help="Model-based non-ink correction threshold. Only applicable if --ink_labels_path is provided.",
    )
    return parser


if __name__ == "__main__":
    main()
