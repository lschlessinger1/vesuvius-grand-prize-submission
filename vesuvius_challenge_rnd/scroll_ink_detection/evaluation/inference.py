import importlib
import logging
import os
from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np
import scipy.stats as st
import torch.nn.functional as F
from dotenv import load_dotenv
from PIL import Image
from pytorch_lightning import LightningModule, Trainer
from scipy.signal.windows import windows

from vesuvius_challenge_rnd import SCROLL_DATA_DIR
from vesuvius_challenge_rnd.data.constants import Z_NON_REVERSED_SEGMENT_IDS, Z_REVERSED_SEGMENT_IDS
from vesuvius_challenge_rnd.scroll_ink_detection import UNet3dSegformerPLModel
from vesuvius_challenge_rnd.scroll_ink_detection.evaluation.util import (
    validate_file_path,
    validate_positive_int,
)
from vesuvius_challenge_rnd.scroll_ink_detection.ink_detection import ScrollPatchDataModuleEval

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

DEFAULT_OUTPUT_DIR = Path("outputs")


def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel."""
    x = np.linspace(-nsig, nsig, kernlen + 1)
    kern1d = np.diff(st.norm.cdf(x))
    kern2d = np.outer(kern1d, kern1d)
    return kern2d / kern2d.sum()


def process_predictions(
    predictions,
    pred_shape: tuple[int, int],
    size: int,
    orig_h: int,
    orig_w: int,
    scale_factor: int = 1,
    window_type: str = "hann",
):
    """
    Processes a list of predictions and their corresponding patch positions.

    :param predictions: List of tuples, each containing a tensor of predictions and a tensor of patch positions.
    :param pred_shape: Tuple indicating the shape of the full prediction area.
    :param size: Size of the square patch for each prediction.
    :param window_type: The window type to use for the predictions patches.
    :return: A numpy array representing the processed prediction mask.
    """

    # Initialize masks
    mask_pred = np.zeros(pred_shape)
    if window_type == "hann":
        kernel = windows.hann(size)[:, None] * windows.hann(size)[None, :]
    elif window_type == "gaussian":
        kernel = gkern(size, 1)
        kernel /= kernel.max()
    else:
        raise ValueError(f"Unknown window {window_type}. Must be either 'hann' or 'gaussian'.")

    # Iterate over the predictions and positions
    for y_proba, patch_position in predictions:
        xys = patch_position.numpy()
        for i, (x1, y1, x2, y2) in enumerate(xys):
            interpolated_pred = (
                F.interpolate(
                    y_proba[i].unsqueeze(0).float(), scale_factor=scale_factor, mode="bilinear"
                )
                .squeeze(0)
                .squeeze(0)
                .numpy()
            )
            y_proba_patch = np.multiply(interpolated_pred, kernel)
            mask_pred[y1:y2, x1:x2] += y_proba_patch

    # Finalize the prediction mask
    mask_pred /= mask_pred.max()
    mask_pred = np.clip(np.nan_to_num(mask_pred), a_min=0, a_max=1)
    mask_pred = (mask_pred * 255).astype(np.uint8)
    mask_pred = mask_pred[:orig_h, :orig_w]
    return mask_pred


def run_inference(
    prediction_segment_id: str,
    scroll_id: str,
    model_ckpt_path: str,
    stride: int,
    z_start: int,
    z_extent: int,
    size: int,
    batch_size: int = 512,
    num_workers: int = 0,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    z_reverse: bool = False,
    infer_z_reversal: bool = False,
    data_dir: Path = SCROLL_DATA_DIR,
    model_cls: type[LightningModule] | str = UNet3dSegformerPLModel,
    skip_if_exists: bool = True,
    scale_factor: int = 1,
    window_type: str = "hann",
):
    if isinstance(model_cls, str):
        model_cls_path = (
            f"vesuvius_challenge_rnd.scroll_ink_detection.ink_detection.lit_models.{model_cls}"
        )
        module_path, class_name = model_cls_path.rsplit(".", 1)
        model_cls = getattr(importlib.import_module(module_path), class_name)

    # Possibly infer the z-reversed flag.
    if infer_z_reversal:
        segment_name_base_part = prediction_segment_id.split("_C")[0].split("_superseded")[0]
        if segment_name_base_part in Z_REVERSED_SEGMENT_IDS:
            z_reverse_inferred = True
        elif segment_name_base_part in Z_NON_REVERSED_SEGMENT_IDS:
            z_reverse_inferred = False
        else:
            z_reverse_inferred = None
            logger.warning(
                f"Unable to infer z-reversal for segment {prediction_segment_id}. Defaulting to given z-reverse ({z_reverse})."
            )

        if (
            z_reverse_inferred is not None
            and isinstance(z_reverse_inferred, bool)
            and z_reverse_inferred != z_reverse
        ):
            logger.info(
                f"Inferred z-reversal ({infer_z_reversal}) different from given z-reversal ({z_reverse}). Using z-reverse={z_reverse_inferred}."
            )
            z_reverse = z_reverse_inferred

    mask_pred_stem = f"{prediction_segment_id}_{stride}_{z_start}_r{int(z_reverse)}"
    mask_pred_path = output_dir / f"{mask_pred_stem}.png"
    if mask_pred_path.exists() and skip_if_exists:
        logger.info(
            f"Skipping segment {prediction_segment_id} prediction because mask prediction path ({mask_pred_path}) already exists."
        )
        return

    # Load model
    model = model_cls.load_from_checkpoint(model_ckpt_path, strict=False)

    # Load data
    data_module = ScrollPatchDataModuleEval(
        prediction_segment_id=prediction_segment_id,
        scroll_id=scroll_id,
        data_dir=data_dir,
        z_min=z_start,
        z_max=z_start + z_extent,
        size=size,
        z_reverse=z_reverse,
        tile_size=size,
        patch_stride=stride,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    # Predict
    trainer = Trainer(precision="16-mixed", devices=1)
    predictions = trainer.predict(model=model, datamodule=data_module)

    orig_h, orig_w = cv2.imread(
        f"{data_module.segment_dir}/{prediction_segment_id}/{prediction_segment_id}_mask.png", 0
    ).shape
    pad0 = size - orig_h % size
    pad1 = size - orig_w % size
    mask_pred = process_predictions(
        predictions,
        (orig_h + pad0, orig_w + pad1),
        data_module.size,
        orig_h,
        orig_w,
        scale_factor=scale_factor,
        window_type=window_type,
    )

    # Save predictions
    output_dir.mkdir(exist_ok=True, parents=True)
    mask_pred = Image.fromarray(mask_pred)
    mask_pred.save(mask_pred_path)
    logger.info(f"Saved prediction mask to {mask_pred_path.resolve()}")


def _set_up_parser() -> ArgumentParser:
    """Set up argument parser for command-line interface.

    Returns:
        ArgumentParser: Configured argument parser.
    """
    parser = ArgumentParser(description="Visualize a patch model's predictions on scroll segments.")
    parser.add_argument("ckpt_path", type=validate_file_path, help="Model checkpoint path")
    parser.add_argument("prediction_segment_id", type=str, help="Prediction segment ID")
    parser.add_argument(
        "--scroll_id", type=str, default="1", help="The scroll ID for which the segment belongs"
    )
    parser.add_argument(
        "-o", "--output_dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Output path"
    )
    parser.add_argument(
        "--data_dir", type=str, default=SCROLL_DATA_DIR, help="Scroll data directory"
    )
    parser.add_argument(
        "-s", "--patch_stride", default=None, type=validate_positive_int, help="Patch stride"
    )
    parser.add_argument(
        "--z_min",
        default=15,
        type=validate_positive_int,
        help="Minimum z-coordinate.",
    )
    parser.add_argument(
        "--z_extent",
        default=32,
        type=validate_positive_int,
        help="Z-coordinate extent.",
    )
    parser.add_argument("--z_reverse", default=False, action="store_true", help="Z-reverse flag")
    parser.add_argument(
        "--stride",
        default=8,
        type=validate_positive_int,
        help="XY-plane stride.",
    )
    parser.add_argument(
        "--size",
        default=64,
        type=validate_positive_int,
        help="Patch XY size.",
    )
    parser.add_argument(
        "--batch_size",
        default=320,
        type=validate_positive_int,
        help="Batch size.",
    )
    parser.add_argument(
        "--num_workers",
        default=4,
        type=int,
        help="Number of workers.",
    )
    parser.add_argument(
        "--skip_if_exists",
        action="store_false",
        default=True,
        help="Skip segment if it already has predictions.",
    )
    parser.add_argument(
        "--infer_z_reversal",
        action="store_true",
        default=False,
        help="Infer the z-reversal setting.",
    )
    parser.add_argument(
        "--window_type",
        default="hann",
        type=str,
        help="The window type to apply on patch predictions.",
    )
    parser.add_argument(
        "--scale_factor",
        default=1,
        type=int,
        help="The label upscaling factor.",
    )
    parser.add_argument(
        "--model_cls_type",
        default="UNet3dSegformerPLModel",
        type=str,
        help="The pytorch lightning class path.",
    )
    return parser


if __name__ == "__main__":
    load_dotenv()
    parser = _set_up_parser()
    args = parser.parse_args()
    run_inference(
        args.prediction_segment_id,
        args.scroll_id,
        args.ckpt_path,
        data_dir=Path(args.data_dir),
        stride=args.stride,
        z_start=args.z_min,
        z_extent=args.z_extent,
        size=args.size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        z_reverse=args.z_reverse,
        infer_z_reversal=args.infer_z_reversal,
        output_dir=Path(args.output_dir),
        skip_if_exists=args.skip_if_exists,
        window_type=args.window_type,
        scale_factor=args.scale_factor,
        model_cls=args.model_cls_type,
    )
