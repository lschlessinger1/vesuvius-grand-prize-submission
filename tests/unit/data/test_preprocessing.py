import numpy as np

from vesuvius_challenge_rnd.data import preprocess_subvolume


def test_preprocess_subvolume():
    input_arr = np.ones((10, 5))
    output = preprocess_subvolume(input_arr)
    assert isinstance(output, np.ndarray)
    assert output.shape == input_arr.shape
    assert output.dtype == np.float32
