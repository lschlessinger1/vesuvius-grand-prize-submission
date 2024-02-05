import pytest
import torch

from vesuvius_challenge_rnd.fragment_ink_detection.ink_detection.models import UNet3Dto2D


@pytest.fixture
def unet_3d_to_2d() -> UNet3Dto2D:
    return UNet3Dto2D()


def test_unet_3d_to_2d_forward(unet_3d_to_2d):
    unet_3d_to_2d.eval()
    input_sample = torch.randn(2, 1, 10, 100, 80)
    with torch.no_grad():
        output = unet_3d_to_2d(input_sample)
    logits = output.logits
    assert isinstance(logits, torch.Tensor)
    assert logits.ndim == 4
    assert logits.shape[0] == 2
    assert logits.shape[1] == 1
    assert logits.shape[2] == 100
    assert logits.shape[3] == 80
