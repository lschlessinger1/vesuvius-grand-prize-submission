import os

import numpy as np
import pytest

from vesuvius_challenge_rnd import SCROLL_DATA_DIR
from vesuvius_challenge_rnd.data.ppm import PPM

# Scroll data is required for these tests.
pytestmark = pytest.mark.scroll_data


@pytest.fixture
def ppm_path() -> str:
    scroll_id = "1"
    segment_name = "20230504125349"
    volume_dir_path = SCROLL_DATA_DIR / str(scroll_id) / segment_name
    ppm_local_path = volume_dir_path / f"{segment_name}.ppm"
    return str(ppm_local_path)


@pytest.fixture
def ppm(ppm_path: str) -> PPM:
    return PPM.from_path(ppm_path)


def test_load_ppm(ppm: PPM):
    assert ppm.is_loaded()
    assert ppm.dim == 6
    assert isinstance(ppm.data, np.ndarray)
    assert ppm.data.ndim == 3
    assert ppm.data.shape[2] == 6


def test_ppm_shape(ppm: PPM):
    # FIXME: Only works for segment: 20230504125349
    assert ppm.width == 556
    assert ppm.height == 652
    assert ppm.dim == 6
    assert ppm.ordered
    assert ppm.type == "double"
    assert ppm.version == "1"

    # Test data
    assert ppm.data.shape == (652, 556, 6)

    # Test some random indices.
    np.testing.assert_array_equal(
        ppm.data[50, 50],
        np.array(
            [
                3891.8581099869857,
                2473.7598214240725,
                27.339080890883583,
                -0.7284660659909741,
                -0.552310577979698,
                -0.4053272950978821,
            ]
        ),
    )
    np.testing.assert_array_equal(ppm.data[0, 0], np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    np.testing.assert_array_equal(ppm.data[-1, -1], np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
