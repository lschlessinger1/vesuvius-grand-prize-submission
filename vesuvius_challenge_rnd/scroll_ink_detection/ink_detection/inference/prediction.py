import numpy as np

from vesuvius_challenge_rnd.fragment_ink_detection import PatchLitModel
from vesuvius_challenge_rnd.fragment_ink_detection.ink_detection.inference.patch_aggregation import (
    parse_predictions_without_labels,
    patches_to_y_proba,
)
from vesuvius_challenge_rnd.fragment_ink_detection.ink_detection.inference.prediction import (
    get_predictions,
)
from vesuvius_challenge_rnd.scroll_ink_detection.ink_detection.data.eval_scroll_patch_data_module import (
    EvalScrollPatchDataModule,
)


def predict_with_model_on_scroll(
    lit_model: PatchLitModel,
    data_module: EvalScrollPatchDataModule,
    patch_surface_shape: tuple[int, int],
    prog_bar: bool = True,
) -> np.ndarray:
    """Predict with a patch model on a given scroll.

    This function retrieves predictions from the model and aggregates them into a smooth probability map
    for the entire scroll segment.

    Args:
        lit_model (PatchLitModel): Trained model to make predictions.
        data_module (EvalScrollPatchDataModule): Data module containing evaluation data for the scroll.
        patch_surface_shape (tuple[int, int]): Shape of the patch surface.
        prog_bar (bool, optional): Whether to display a progress bar. Defaults to True.

    Returns:
        np.ndarray: The smoothed probability map for the scroll.
    """
    predictions = get_predictions(lit_model, data_module, prog_bar)

    y_proba_patches, patch_positions = parse_predictions_without_labels(predictions)

    # Aggregate patch predictions.
    mask = data_module.data_predict.masks[0]
    y_proba_smoothed = patches_to_y_proba(
        y_proba_patches, patch_positions, mask, patch_surface_shape
    )

    return y_proba_smoothed
