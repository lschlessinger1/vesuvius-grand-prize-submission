from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv
from matplotlib import pyplot as plt
from pytorch_lightning.core.saving import load_hparams_from_yaml
from tqdm.auto import tqdm

from vesuvius_challenge_rnd.fragment_ink_detection import EvalPatchDataModule, PatchLitModel
from vesuvius_challenge_rnd.fragment_ink_detection.ink_detection.inference.prediction import (
    predict_with_model_on_fragment,
)


def visualize_predictions(
    lit_model: PatchLitModel,
    patch_surface_shape: tuple[int, int],
    z_min: int = 27,
    z_max: int = 37,
    downsampling: int | None = None,
    patch_stride: int | None = None,
    predict_fragment_ind: list[int] | None = None,
    batch_size: int = 4,
    num_workers: int = 0,
    thresh: float = 0.5,
    save_dir: Path | None = Path("outputs"),
) -> None:
    """Visualize predictions on a given model.

    Args:
        lit_model (PatchLitModel): The model for predictions.
        patch_surface_shape (tuple[int, int]): Shape of the surface patch.
        z_min (int, optional): Minimum z-value. Defaults to 27.
        z_max (int, optional): Maximum z-value. Defaults to 37.
        downsampling (int | None, optional): Downsampling factor. Defaults to None.
        patch_stride (int | None, optional): Stride of the patch. Defaults to None.
        predict_fragment_ind (list[int] | None, optional): Indices for prediction fragment. Defaults to None.
        batch_size (int, optional): Batch size for predictions. Defaults to 4.
        num_workers (int, optional): Number of workers for parallelism. Defaults to 0.
        thresh (float, optional): Threshold for predictions. Defaults to 0.5.
        save_dir (Path | None, optional): Directory to save outputs. Defaults to Path("outputs").
    """
    if predict_fragment_ind is None:
        predict_fragment_ind = [1, 2, 3]

    if patch_stride is None:
        patch_stride = patch_surface_shape[0] // 2

    for index in tqdm(predict_fragment_ind):
        # Initialize data module.
        data_module = EvalPatchDataModule(
            predict_fragment_ind=[index],
            z_min=z_min,
            z_max=z_max,
            patch_surface_shape=patch_surface_shape,
            patch_stride=patch_stride,
            downsampling=downsampling,
            num_workers=num_workers,
            batch_size=batch_size,
        )

        # Get and parse predictions.
        y_proba_smoothed = predict_with_model_on_fragment(
            lit_model, data_module, patch_surface_shape
        )

        # Show predictions next to ground truth.
        mask = data_module.data_predict.masks[0]
        ink_labels = data_module.data_predict.labels[0]
        ir_img = data_module.data_predict.fragments[0].load_ir_img()

        fig, ax = create_pred_fig(index, thresh, mask, y_proba_smoothed, ink_labels, ir_img)

        if save_dir is not None:
            # Optionally save the figure.
            save_dir.mkdir(exist_ok=True)
            file_name = f"prediction_{index}.png"
            output_path = save_dir / file_name
            fig.savefig(output_path)
            print(f"Saved prediction to {output_path.resolve()}")

        plt.show()


def create_pred_fig(
    index: int,
    thresh: float,
    mask: np.ndarray,
    y_proba_smoothed: np.ndarray,
    ink_labels: np.ndarray,
    ir_img: np.ndarray,
    figsize: tuple[int, int] = (25, 10),
) -> tuple[plt.Figure, plt.Axes]:
    """Create a figure with predictions, ink labels, IR image, and ink prediction.

    Args:
        index (int): Fragment index.
        thresh (float): Threshold for predictions.
        mask (np.ndarray): Mask array.
        y_proba_smoothed (np.ndarray): Smoothed probability array.
        ink_labels (np.ndarray): Ink label array.
        ir_img (np.ndarray): Infrared image array.
        figsize (tuple[int, int], optional): Figure size. Defaults to (25, 10).

    Returns:
        tuple[plt.Figure, plt.Axes]: Figure and Axes objects for the plot.
    """
    fig, ax = plt.subplots(1, 4, figsize=figsize)
    ax1, ax2, ax3, ax4 = ax
    fig.suptitle(f"Fragment {index}")

    ax1.set_title("Predictions")
    ax1.imshow(mask)

    im2 = ax1.imshow(y_proba_smoothed)
    plt.colorbar(im2, ax=ax1)
    ax2.set_title("Ink labels")
    ax2.imshow(ink_labels, cmap="binary")

    ax3.imshow(ir_img, cmap="gray")
    ax3.set_title("IR image")

    ax4.imshow(y_proba_smoothed > thresh, cmap="binary")
    ax4.set_title(f"Ink prediction (thresh={thresh})")

    return fig, ax


def load_model(ckpt_path: str, map_location: torch.device | None = None) -> PatchLitModel:
    """Load a model from a checkpoint.

    Args:
        ckpt_path (str): Checkpoint path.
        map_location (torch.device | None, optional): Device mapping location. Defaults to None.

    Returns:
        PatchLitModel: Loaded model.
    """
    # Initialize model.
    lit_model = PatchLitModel.load_from_checkpoint(ckpt_path, map_location=map_location)
    lit_model.eval()
    return lit_model


def parse_config(config_path: str | Path) -> dict:
    """Parse the configuration from a given YAML file.

    Args:
        config_path (str): Path to the configuration file.

    Returns:
        dict: Dictionary containing configuration parameters.
    """
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Config not found at path {config_path}.")

    config = load_hparams_from_yaml(config_path)

    if len(config) == 0:
        raise ValueError(f"Found empty config from path: {config_path}")

    patch_surface_shape = config["data"]["init_args"]["patch_surface_shape"]
    z_min = config["data"]["init_args"]["z_min"]
    z_max = config["data"]["init_args"]["z_max"]
    downsampling = config["data"]["init_args"]["downsampling"]
    return {
        "patch_surface_shape": patch_surface_shape,
        "z_min": z_min,
        "z_max": z_max,
        "downsampling": downsampling,
    }


def main(
    ckpt_path: str,
    config_path: str,
    patch_stride: int | None = None,
    pred_frag_ind: list[int] | None = None,
) -> None:
    """Main function to run the visualization.

    Args:
        ckpt_path (str): Model checkpoint path.
        config_path (str): Training configuration path.
        patch_stride (int | None, optional): Patch stride. Defaults to None.
        pred_frag_ind (list[int] | None, optional): Prediction fragment indices. Defaults to None.
    """
    data_config = parse_config(config_path)
    lit_model = load_model(ckpt_path)
    print(f"patch_stride: {patch_stride}")
    print(f"pred_frag_ind: {pred_frag_ind}")
    visualize_predictions(
        lit_model, patch_stride=patch_stride, predict_fragment_ind=pred_frag_ind, **data_config
    )


def _set_up_parser() -> ArgumentParser:
    """Set up argument parser for command-line interface.

    Returns:
        ArgumentParser: Configured argument parser.
    """
    parser = ArgumentParser(description="Visualize a patch model's predictions on fragments.")
    parser.add_argument("ckpt_path", type=str, help="Model checkpoint path")
    parser.add_argument("cfg_path", type=str, help="Training config path")
    parser.add_argument("-s", "--patch_stride", default=None, type=int, help="Patch stride")
    parser.add_argument(
        "-f", "--pred_frag_ind", nargs="+", type=int, help="Prediction fragment indices"
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
        pred_frag_ind=args.pred_frag_ind,
    )
