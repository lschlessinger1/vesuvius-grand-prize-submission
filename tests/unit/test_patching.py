from vesuvius_challenge_rnd.patching import patch_index_to_pixel_position


def test_patch_index_to_pixel_position():
    (y0, x0), (y1, x1) = patch_index_to_pixel_position(0, 0, (500, 600), 100)
    assert y0 == 0
    assert x0 == 0
    assert y1 == 500
    assert x1 == 600
