"""Single-model prediction."""
import numpy as np
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks import RichProgressBar

from vesuvius_challenge_rnd.fragment_ink_detection import EvalPatchDataModule, PatchLitModel
from vesuvius_challenge_rnd.fragment_ink_detection.ink_detection.inference.patch_aggregation import (
    parse_predictions_without_labels,
    patches_to_y_proba,
)


def get_predictions(
    lit_model: LightningModule, data_module: LightningDataModule, prog_bar: bool = True
) -> list:
    """Retrieve predictions using the given model and data module.

    Args:
        lit_model (LightningModule): Trained model to make predictions.
        data_module (LightningDataModule): Data module containing evaluation data.
        prog_bar (bool, optional): Whether to display a progress bar. Defaults to True.

    Returns:
        list: List of predictions made by the model.
    """
    callbacks = []

    if prog_bar:
        callbacks.append(RichProgressBar())
    trainer = Trainer(callbacks=callbacks)
    predictions = trainer.predict(lit_model, datamodule=data_module)
    return predictions


def predict_with_model_on_fragment(
    lit_model: PatchLitModel,
    data_module: EvalPatchDataModule,
    patch_surface_shape: tuple[int, int],
    prog_bar: bool = True,
) -> np.ndarray:
    """Predict with a patch model on a given fragment.

    This function retrieves predictions from the model and aggregates them into a smooth probability map
    for the entire fragment.

    Args:
        lit_model (PatchLitModel): Trained model to make predictions.
        data_module (EvalPatchDataModule): Data module containing evaluation data for the fragment.
        patch_surface_shape (tuple[int, int]): Shape of the patch surface.
        prog_bar (bool, optional): Whether to display a progress bar. Defaults to True.

    Returns:
        np.ndarray: The smoothed probability map for the fragment.
    """
    predictions = get_predictions(lit_model, data_module, prog_bar)

    y_proba_patches, patch_positions = parse_predictions_without_labels(predictions)

    # Aggregate patch predictions.
    mask = data_module.data_predict.masks[0]
    y_proba_smoothed = patches_to_y_proba(
        y_proba_patches, patch_positions, mask, patch_surface_shape
    )

    return y_proba_smoothed
