from pathlib import Path

import numpy as np
import pytest

from vesuvius_challenge_rnd.data import Scroll, ScrollSegment

# Scroll data is required for these tests.
pytestmark = pytest.mark.scroll_data


@pytest.fixture(
    params=[
        ("1", "20230504093154"),
        ("2", "20230421192746"),
    ]
)
def scroll_segment(request) -> ScrollSegment:
    scroll_id, segment_name = request.param
    yield ScrollSegment(scroll_id, segment_name)


@pytest.fixture(
    params=[
        "1",
        "2",
    ]
)
def scroll(request) -> Scroll:
    scroll_id = request.param
    yield Scroll(scroll_id)


def test_scroll_segment_init(scroll_segment: ScrollSegment):
    assert scroll_segment.data_dir.exists()
    assert scroll_segment.volume_dir_path.exists()
    assert isinstance(scroll_segment.segment_name, str)
    assert isinstance(scroll_segment.scroll_id, str)
    assert isinstance(scroll_segment.surface_volume_dir_name, str)


def test_scroll_segment_load_surface_vol_paths(scroll_segment: ScrollSegment):
    paths = scroll_segment.load_surface_vol_paths()
    assert all(isinstance(path, Path) for path in paths)
    assert all(path.exists() for path in paths)


def test_scroll_segment_load_volume_single_slice(scroll_segment: ScrollSegment):
    img_stack = scroll_segment.load_volume(z_start=27, z_end=28)
    assert isinstance(img_stack, np.ndarray)
    assert img_stack.dtype == np.float32
    assert img_stack.shape == scroll_segment.surface_shape


def test_scroll_segment_load_volume_as_memmap_single_slice(scroll_segment: ScrollSegment):
    img_stack = scroll_segment.load_volume_as_memmap(z_start=27, z_end=28)
    assert isinstance(img_stack, np.memmap)
    assert img_stack.shape == (1, *scroll_segment.surface_shape)


def test_scroll_segment_load_mask(scroll_segment: ScrollSegment):
    mask = scroll_segment.load_mask()
    assert isinstance(mask, np.ndarray)
    assert mask.dtype == bool
    assert mask.shape == scroll_segment.surface_shape


def test_scroll_segment_shape(scroll_segment: ScrollSegment):
    shape = scroll_segment.shape
    assert isinstance(shape, tuple)
    assert len(shape) == 3
    assert all(isinstance(d, int) for d in shape)


def test_scroll_segment_n_slices(scroll_segment: ScrollSegment):
    n_slices = scroll_segment.n_slices
    assert isinstance(n_slices, int)
    assert n_slices == 65


def test_scroll_segment_surface_shape(scroll_segment: ScrollSegment):
    shape = scroll_segment.surface_shape
    assert isinstance(shape, tuple)
    assert len(shape) == 2
    assert all(isinstance(d, int) for d in shape)


def test_fragment_voxel_size_microns(scroll_segment: ScrollSegment):
    voxel_size = scroll_segment.voxel_size_microns
    assert isinstance(voxel_size, float)
    assert voxel_size > 0


def test_scroll_segment_author(scroll_segment: ScrollSegment):
    assert isinstance(scroll_segment.author, str)


def test_scroll_segment_area_cm2(scroll_segment: ScrollSegment):
    area_cm2 = scroll_segment.area_cm2
    assert isinstance(area_cm2, float)
    assert area_cm2 > 0


def test_scroll_init(scroll: Scroll):
    assert len(scroll.segments) > 0
    assert isinstance(scroll.segments[0], ScrollSegment)


def test_load_ppm():
    segment = ScrollSegment("1", "20230504125349")
    ppm = segment.load_ppm()

    assert ppm.is_loaded()
    assert ppm.dim == 6
    assert isinstance(ppm.data, np.ndarray)
