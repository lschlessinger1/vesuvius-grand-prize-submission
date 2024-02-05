from pathlib import Path

import numpy as np
import pytest

from vesuvius_challenge_rnd import FRAGMENT_DATA_DIR
from vesuvius_challenge_rnd.data import Fragment

# Fragment data is required for these tests.
pytestmark = pytest.mark.fragment_data


@pytest.fixture(params=list(FRAGMENT_DATA_DIR.glob("*")))
def available_fragment_id(request) -> int:
    yield int(str(request.param.name))


@pytest.fixture
def fragment(available_fragment_id: int) -> Fragment:
    yield Fragment(available_fragment_id)


def test_fragment_init(fragment: Fragment):
    assert fragment.data_dir.exists()
    assert fragment.volume_dir_path.exists()
    assert isinstance(fragment.segment_name, str)
    assert isinstance(fragment.fragment_id, int)
    assert isinstance(fragment.surface_volume_dir_name, str)


def test_fragment_load_surface_vol_paths(fragment: Fragment):
    paths = fragment.load_surface_vol_paths()
    assert all(isinstance(path, Path) for path in paths)
    assert all(path.exists() for path in paths)


def test_fragment_load_volume_single_slice(fragment: Fragment):
    img_stack = fragment.load_volume(z_start=27, z_end=28)
    assert isinstance(img_stack, np.ndarray)
    assert img_stack.dtype == np.float32
    assert img_stack.shape == fragment.surface_shape


def test_fragment_load_volume_as_memmap_single_slice(fragment: Fragment):
    img_stack = fragment.load_volume_as_memmap(z_start=27, z_end=28)
    assert isinstance(img_stack, np.memmap)
    assert img_stack.shape == (1, *fragment.surface_shape)


def test_fragment_load_mask(fragment: Fragment):
    mask = fragment.load_mask()
    assert isinstance(mask, np.ndarray)
    assert mask.dtype == bool
    assert mask.shape == fragment.surface_shape


def test_fragment_shape(fragment: Fragment):
    shape = fragment.shape
    assert isinstance(shape, tuple)
    assert len(shape) == 3
    assert all(isinstance(d, int) for d in shape)


def test_fragment_n_slices(fragment: Fragment):
    n_slices = fragment.n_slices
    assert isinstance(n_slices, int)
    assert n_slices == 65


def test_fragment_surface_shape(fragment: Fragment):
    shape = fragment.surface_shape
    assert isinstance(shape, tuple)
    assert len(shape) == 2
    assert all(isinstance(d, int) for d in shape)


def test_fragment_load_ink_labels(fragment: Fragment):
    ink_labels = fragment.load_ink_labels()
    assert isinstance(ink_labels, np.ndarray)
    assert ink_labels.dtype == bool
    assert ink_labels.shape == fragment.surface_shape


def test_fragment_load_ir_img(fragment: Fragment):
    ir_img = fragment.load_ir_img()
    assert isinstance(ir_img, np.ndarray)
    assert ir_img.dtype == np.uint8
    assert ir_img.shape == fragment.surface_shape


def test_fragment_voxel_size_microns(fragment: Fragment):
    voxel_size = fragment.voxel_size_microns
    assert isinstance(voxel_size, float)
    assert voxel_size > 0
