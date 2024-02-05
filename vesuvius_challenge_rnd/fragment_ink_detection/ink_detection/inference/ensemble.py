import numpy as np

from vesuvius_challenge_rnd.fragment_ink_detection import EvalPatchDataModule, PatchLitModel
from vesuvius_challenge_rnd.fragment_ink_detection.ink_detection.inference.prediction import (
    predict_with_model_on_fragment,
)


def ensemble_prediction(*y_proba_smoothed: np.ndarray) -> np.ndarray:
    """Compute the ensemble prediction by averaging smoothed probabilities.

    Args:
        *y_proba_smoothed (np.ndarray): Smoothed probabilities for each model in the ensemble.

    Returns:
        np.ndarray: The averaged probability array representing the ensemble prediction.
    """
    return np.dstack(y_proba_smoothed).mean(2)


def ensemble_predict_on_fragment(
    lit_models: list[PatchLitModel],
    data_module: EvalPatchDataModule,
    patch_surface_shape: tuple[int, int],
) -> np.ndarray:
    """Perform ensemble prediction on a fragment using multiple trained models.

    Args:
        lit_models (list[PatchLitModel]): List of trained models to use for ensemble prediction.
        data_module (EvalPatchDataModule): Data module containing evaluation data for the fragment.
        patch_surface_shape (tuple[int, int]): Shape of the patch surface.

    Returns:
        np.ndarray: The ensemble prediction for the fragment.
    """
    y_proba_smoothed = [
        predict_with_model_on_fragment(model, data_module, patch_surface_shape)
        for model in lit_models
    ]
    return ensemble_prediction(*y_proba_smoothed)
