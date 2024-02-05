from abc import ABC, abstractmethod

from vesuvius_challenge_rnd.fragment_ink_detection.ink_detection.data.patch_dataset import (
    PatchDataset,
)


class AbstractFragmentValPatchDataset(ABC):
    def __init__(self):
        super().__init__()

    @property
    @abstractmethod
    def val_fragment_dataset(self) -> PatchDataset:
        raise NotImplementedError
