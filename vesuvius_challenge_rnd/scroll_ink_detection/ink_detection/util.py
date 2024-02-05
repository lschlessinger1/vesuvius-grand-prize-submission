import numpy as np


def pad_to_match(a: np.ndarray, b: np.ndarray, pad_value: int = 0) -> np.ndarray:
    """
    Pad the smaller of two arrays 'a' and 'b' so that they have the same shape.
    """
    pad_y = max(a.shape[0], b.shape[0]) - a.shape[0]
    pad_x = max(a.shape[1], b.shape[1]) - a.shape[1]
    return np.pad(a, [(0, pad_y), (0, pad_x)], constant_values=pad_value)
