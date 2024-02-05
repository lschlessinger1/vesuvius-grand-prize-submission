from pathlib import Path

import cv2
import numpy as np


def read_ink_preds(
    segment_name: str, ink_preds_dir: Path, expected_shape: tuple[int, int]
) -> np.ndarray:
    ink_pred_paths = list(p for p in ink_preds_dir.glob("*.png") if p.stem.startswith(segment_name))
    if len(ink_pred_paths) != 1:
        raise ValueError(
            f"Found {len(ink_pred_paths)} ink predictions for {segment_name} in {ink_preds_dir}. Expected one prediction per segment."
        )

    ink_pred_path = ink_pred_paths[0]

    if not ink_pred_path.is_file():
        raise FileNotFoundError(f"Could not find ink prediction: {ink_pred_path.resolve()}.")

    ink_pred = cv2.imread(str(ink_pred_path), cv2.IMREAD_GRAYSCALE)

    # Pad if necessary.
    pad_y = expected_shape[0] - ink_pred.shape[0]
    pad_x = expected_shape[1] - ink_pred.shape[1]
    if pad_x < 0 or pad_y < 0:
        raise ValueError(
            f"expected shape ({expected_shape}) must be larger than raw ink pred shape ({ink_pred.shape}) for all dimensions."
        )
    ink_pred = np.pad(ink_pred, [(0, pad_y), (0, pad_x)], constant_values=0)

    return ink_pred
