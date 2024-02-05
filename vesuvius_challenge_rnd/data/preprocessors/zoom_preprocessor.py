from pathlib import Path

import numpy as np
from scipy.ndimage import zoom

from vesuvius_challenge_rnd.data.preprocessors.fragment_preprocessor import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_ZOOM_FACTOR,
    FragmentPreprocessor,
)


class ZoomPreprocessor(FragmentPreprocessor):
    # FIXME: this needs to be tested
    def __init__(
        self,
        output_dir: Path = DEFAULT_OUTPUT_DIR,
        zoom_factor: float = DEFAULT_ZOOM_FACTOR,
        order: int = 3,
        preprocess_ir_img: bool = False,
        skip_if_exists: bool = True,
    ):
        super().__init__(
            output_dir, preprocess_ir_img=preprocess_ir_img, skip_if_exists=skip_if_exists
        )
        self.zoom_factor = zoom_factor
        self.order = order

    def _transform_surface_volumes(self, surface_volumes: np.ndarray) -> np.ndarray:
        return zoom(
            surface_volumes,
            zoom=(self.zoom_factor, self.zoom_factor, self.zoom_factor),
            order=self.order,
        )

    def _transform_labels(self, labels: np.ndarray) -> np.ndarray:
        return zoom(labels, zoom=(self.zoom_factor, self.zoom_factor), order=self.order)

    def _transform_mask(self, mask: np.ndarray) -> np.ndarray:
        return zoom(mask, zoom=(self.zoom_factor, self.zoom_factor), order=self.order)

    @property
    def method_name(self) -> str:
        return f"zoom__zoom_factor={self.zoom_factor}__order={self.order}"

    def __repr__(self):
        return (
            f"{type(self).__name__}(output_dir={self.output_dir}, preprocess_ir_img={self.preprocess_ir_img}, "
            f"zoom_factor={self.zoom_factor}, order={self.order})"
        )
