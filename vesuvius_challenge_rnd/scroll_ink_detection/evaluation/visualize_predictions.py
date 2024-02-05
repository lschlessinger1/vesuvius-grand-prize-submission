import importlib
import logging
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv
from matplotlib import pyplot as plt
from pytorch_lightning.core.saving import load_hparams_from_yaml
from tqdm.auto import tqdm

from vesuvius_challenge_rnd import SCROLL_DATA_DIR
from vesuvius_challenge_rnd.data import Scroll, ScrollSegment
from vesuvius_challenge_rnd.data.scroll import create_scroll_segment
from vesuvius_challenge_rnd.fragment_ink_detection import PatchLitModel
from vesuvius_challenge_rnd.scroll_ink_detection.evaluation.util import (
    tuple_parser,
    validate_file_path,
    validate_positive_int,
)
from vesuvius_challenge_rnd.scroll_ink_detection.ink_detection.data.eval_scroll_patch_data_module import (
    EvalScrollPatchDataModule,
)
from vesuvius_challenge_rnd.scroll_ink_detection.ink_detection.inference.prediction import (
    predict_with_model_on_scroll,
)

DEFAULT_OUTPUT_DIR = Path("outputs")


def visualize_predictions(
    lit_model: PatchLitModel,
    patch_surface_shape: tuple[int, int],
    z_patch_size: int,
    z_min: int = 27,
    z_max: int = 37,
    z_stride: int = 1,
    downsampling: int | None = None,
    patch_stride: int | None = None,
    predict_scroll_ind: list[int] | None = None,
    segment_data: list[tuple[int, str]] | None = None,
    batch_size: int = 4,
    num_workers: int = 0,
    save_dir: Path | None = DEFAULT_OUTPUT_DIR,
) -> None:
    """Visualize predictions on a given model.

    Args:
        lit_model (PatchLitModel): The model for predictions.
        patch_surface_shape (tuple[int, int]): Shape of the surface patch.
        z_min (int, optional): Minimum z-value. Defaults to 27.
        z_max (int, optional): Maximum z-value. Defaults to 37.
        downsampling (int | None, optional): Downsampling factor. Defaults to None.
        patch_stride (int | None, optional): Stride of the patch. Defaults to None.
        predict_scroll_ind (list[int] | None, optional): Indices for prediction scroll. Defaults to None.
        segment_data (list[tuple[int, str]] | None, optional): Scroll segment data tuples. Defaults to None.
        batch_size (int, optional): Batch size for predictions. Defaults to 4.
        num_workers (int, optional): Number of workers for parallelism. Defaults to 0.
        save_dir (Path | None, optional): Directory to save outputs. Defaults to Path("outputs").
    """
    if patch_stride is None:
        patch_stride = patch_surface_shape[0] // 2

    segments = _get_scroll_segments(predict_scroll_ind, segment_data, patch_surface_shape)

    for segment in tqdm(segments):
        for z_start in range(z_min, z_max - z_patch_size + 1, z_stride):
            z_end = z_start + z_patch_size
            # Initialize scroll segment data module.
            data_module = EvalScrollPatchDataModule(
                segment,
                z_min=z_start,
                z_max=z_end,
                patch_surface_shape=patch_surface_shape,
                patch_stride=patch_stride,
                downsampling=downsampling,
                num_workers=num_workers,
                batch_size=batch_size,
            )

            # Get and parse predictions.
            y_proba_smoothed = predict_with_model_on_scroll(
                lit_model, data_module, patch_surface_shape
            )

            # Show prediction for segment.
            z_mid = (z_end + z_start) // 2
            texture_img = segment.load_volume(z_mid, z_mid + 1)

            fig, ax = create_pred_fig(
                segment.scroll_id,
                segment.segment_name,
                texture_img,
                y_proba_smoothed,
                z_mid,
                z_min=z_start,
                z_max=z_end,
            )

            if save_dir is not None:
                # Optionally save the figure.
                save_dir.mkdir(exist_ok=True, parents=True)
                file_name = f"prediction_{segment.scroll_id}_{segment.segment_name}_z={z_start}-to-{z_end}.png"
                output_path = save_dir / file_name
                fig.savefig(output_path)
                print(f"Saved prediction to {output_path.resolve()}")

        plt.show()


def create_pred_fig(
    scroll_id: int,
    segment_name: str,
    texture_img: np.ndarray,
    y_proba_smoothed: np.ndarray,
    texture_slice_idx: int,
    z_min: int,
    z_max: int,
    figsize: tuple[int, int] = (25, 10),
) -> tuple[plt.Figure, plt.Axes]:
    """Create a figure with predictions and ink prediction.

    Args:
        scroll_id (int): Scroll ID.
        segment_name (str): Scroll segment name.
        texture_img (np.ndarray): Texture image of the segment.
        y_proba_smoothed (np.ndarray): Smoothed probability array.
        figsize (tuple[int, int], optional): Figure size. Defaults to (25, 10).

    Returns:
        tuple[plt.Figure, plt.Axes]: Figure and Axes objects for the plot.
    """
    horizontal = texture_img.shape[1] > texture_img.shape[0]
    n_plots = 2
    if horizontal:
        n_rows = 1
        n_cols = n_plots
    else:
        n_rows = n_plots
        n_cols = 1

    fig, ax = plt.subplots(n_rows, n_cols, figsize=figsize)
    ax1, ax2 = ax.flatten()
    fig.suptitle(f"Scroll {scroll_id} - segment {segment_name} - z={z_min}-{z_max}")

    ax1.imshow(texture_img, cmap="gray", vmin=0, vmax=1)
    ax1.set_title(f"Slice {texture_slice_idx}")

    ax2.imshow(y_proba_smoothed, cmap="viridis", vmin=0, vmax=1)
    ax2.set_title(f"Ink prediction")

    return fig, ax


def load_model(
    ckpt_path: str,
    model_cls_path: str,
    map_location: torch.device | None = None,
    strict: bool = True,
    **kwargs,
) -> PatchLitModel:
    """Load a model from a checkpoint.

    Args:
        ckpt_path (str): Checkpoint path.
        map_location (torch.device | None, optional): Device mapping location. Defaults to None.

    Returns:
        PatchLitModel: Loaded model.
    """
    # Initialize model.
    module_path, class_name = model_cls_path.rsplit(".", 1)
    lit_model_cls = getattr(importlib.import_module(module_path), class_name)
    lit_model = lit_model_cls.load_from_checkpoint(
        ckpt_path, map_location=map_location, strict=strict, **kwargs
    )
    lit_model.eval()
    return lit_model


def parse_config(config_path: str | Path) -> dict:
    """Parse the configuration from a given YAML file.

    Args:
        config_path (str): Path to the configuration file.

    Returns:
        dict: Dictionary containing configuration parameters.
    """
    config = load_hparams_from_yaml(config_path)
    data_init_args = config["data"]["init_args"]
    patch_surface_shape = data_init_args["patch_surface_shape"]

    if "z_min_scroll" in data_init_args:
        z_min = data_init_args["z_min_scroll"]
    else:
        z_min = data_init_args["z_min"]
    if "z_max_scroll" in data_init_args:
        z_max = data_init_args["z_max_scroll"]
    else:
        z_max = data_init_args["z_max"]
    downsampling = data_init_args["downsampling"]
    model_cls_path = config["model"]["class_path"]
    return {
        "patch_surface_shape": patch_surface_shape,
        "z_min": z_min,
        "z_max": z_max,
        "downsampling": downsampling,
        "model_cls_path": model_cls_path,
    }


def _get_scroll_segments(
    predict_scroll_ind: list[int] | None,
    segment_data: list[tuple[int, str]] | None,
    patch_surface_shape: tuple[int, int],
    scroll_dir: Path = SCROLL_DATA_DIR,
) -> list[ScrollSegment]:
    # Create scroll segments.
    if predict_scroll_ind is not None:
        scrolls = [Scroll(i) for i in predict_scroll_ind]
        segments = [segment for scroll in scrolls for segment in scroll]
    else:
        segments = [
            create_scroll_segment(scroll_id, segment_name, scroll_dir=scroll_dir)
            for scroll_id, segment_name in segment_data
        ]

    # Discard segments that are smaller than the patch size.
    filtered_segments = []
    for segment in segments:
        if (
            segment.surface_shape[0] >= patch_surface_shape[0]
            and segment.surface_shape[1] >= patch_surface_shape[1]
        ):
            filtered_segments.append(segment)
        else:
            logging.warning(
                f"Skipping scroll {segment.scroll_id} segment {segment.segment_name} with shape {segment.shape} because "
                f"it's smaller than the patch surface shape: {patch_surface_shape}."
            )
    return filtered_segments


def main(
    ckpt_path: str,
    config_path: str,
    patch_stride: int | None = None,
    predict_scroll_ind: list[int] | None = None,
    segment_data: list[tuple[int, str]] | None = None,
    output_dir: Path | None = DEFAULT_OUTPUT_DIR,
    z_min: int | None = None,
    z_max: int | None = None,
    z_stride: int = 1,
) -> None:
    """Main function to run the visualization.

    Args:
        ckpt_path (str): Model checkpoint path.
        config_path (str): Training configuration path.
        patch_stride (int | None, optional): Patch stride. Defaults to None.
        predict_scroll_ind (list[int] | None, optional): Prediction scroll indices. Defaults to None.
        segment_data (list[tuple[int, str]] | None, optional): Scroll segment data tuples. Defaults to None.
        output_dir (Path | None, optional): The output directory.
    """
    data_config = parse_config(config_path)

    # Parse z min and max.
    z_extent_orig = data_config["z_max"] - data_config["z_min"]
    if z_min is not None:
        data_config["z_min"] = z_min
    if z_max is not None:
        data_config["z_max"] = z_max
    z_extent = data_config["z_max"] - data_config["z_min"]
    if z_extent < z_extent_orig:
        raise ValueError(
            f"Z-extent ({z_extent}) cannot be smaller than {z_extent_orig} as found in the config."
        )

    model_cls_path = data_config.pop("model_cls_path")
    lit_model = load_model(ckpt_path, model_cls_path=model_cls_path)

    if predict_scroll_ind is None:
        if segment_data is None:  # If both are not set, predict on all scrolls.
            predict_scroll_ind = [1, 2]
    else:
        if segment_data is not None:
            raise ValueError("`predict_scroll_ind` and `segment_data` cannot both be set.")

    visualize_predictions(
        lit_model,
        patch_stride=patch_stride,
        predict_scroll_ind=predict_scroll_ind,
        segment_data=segment_data,
        save_dir=output_dir,
        z_stride=z_stride,
        z_patch_size=z_extent_orig,
        **data_config,
    )


def _set_up_parser() -> ArgumentParser:
    """Set up argument parser for command-line interface.

    Returns:
        ArgumentParser: Configured argument parser.
    """
    parser = ArgumentParser(description="Visualize a patch model's predictions on scroll segments.")
    parser.add_argument("ckpt_path", type=validate_file_path, help="Model checkpoint path")
    parser.add_argument("cfg_path", type=validate_file_path, help="Training config path")
    parser.add_argument(
        "-o", "--output_dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Output path"
    )
    parser.add_argument(
        "-s", "--patch_stride", default=None, type=validate_positive_int, help="Patch stride"
    )
    parser.add_argument(
        "--z_min",
        default=None,
        type=validate_positive_int,
        help="Z-min (defaults to same as used in the lit model config).",
    )
    parser.add_argument(
        "--z_max",
        default=None,
        type=validate_positive_int,
        help="Z-max (defaults to same as used in the lit model config).",
    )
    parser.add_argument(
        "--z_stride",
        default=1,
        type=validate_positive_int,
        help="Z stride. Only used if given a z-range larger than patch depth size.",
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "-f", "--predict_scroll_ind", nargs="+", type=int, help="Prediction scroll indices"
    )
    group.add_argument(
        "-g",
        "--segments",
        nargs="+",
        type=tuple_parser,
        help="Tuples of 'scroll,segment' (e.g., 2,20230421204550)",
    )

    return parser


if __name__ == "__main__":
    load_dotenv()
    parser = _set_up_parser()
    args = parser.parse_args()
    main(
        args.ckpt_path,
        args.cfg_path,
        patch_stride=args.patch_stride,
        predict_scroll_ind=args.predict_scroll_ind,
        z_min=args.z_min,
        z_max=args.z_max,
        z_stride=args.z_stride,
        segment_data=args.segments,
        output_dir=Path(args.output_dir),
    )
