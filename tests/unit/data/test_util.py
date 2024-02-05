import numpy as np

from vesuvius_challenge_rnd.data.util import indices_to_microns, microns_to_indices


def test_microns_to_indices_single_index():
    microns = 5.0
    voxel_size_microns = 2.5
    expected_output = 2
    output = microns_to_indices(microns, voxel_size_microns=voxel_size_microns)
    assert output == expected_output


def test_microns_to_indices_multi_ind():
    microns = [5.0, 10.0, 15.0]
    voxel_size_microns = 2.5
    expected_output = [2, 4, 6]
    output = microns_to_indices(microns, voxel_size_microns=voxel_size_microns)
    assert np.array_equal(output, expected_output)


def test_indices_to_microns_single_index():
    voxel_size_microns = 5.0
    indices = 2
    microns = indices_to_microns(indices, voxel_size_microns)
    assert microns == 10


def test_indices_to_microns_multi_ind():
    indices = [2, 4, 6]
    voxel_size_microns = 2.5
    expected_output = [5.0, 10.0, 15.0]
    output = indices_to_microns(indices, voxel_size_microns=voxel_size_microns)
    assert np.array_equal(output, expected_output)
