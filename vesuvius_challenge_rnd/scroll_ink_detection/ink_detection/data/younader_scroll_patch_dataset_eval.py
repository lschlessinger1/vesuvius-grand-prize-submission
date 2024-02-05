from typing import Optional

from collections.abc import Callable

from torch.utils.data import Dataset


class CustomDatasetTest(Dataset):
    def __init__(self, images, xyxys, transform: Callable | None = None):
        self.images = images
        self.xyxys = xyxys
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        image = self.images[idx]
        xy = self.xyxys[idx]
        if self.transform:
            data = self.transform(image=image)
            image = data["image"].unsqueeze(0)

        return image, xy
