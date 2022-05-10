from dataclasses import asdict
from typing import Dict, List, Callable, Optional

import torch
from PIL import Image

from torch.utils.data import Dataset
from torchvision.transforms import transforms

from common import INCEPTION_INPUT_SIZE, CUBDNDataItem


DUMMY_IMG_TENSOR_SHAPE = (3, INCEPTION_INPUT_SIZE, INCEPTION_INPUT_SIZE)


class CUBDNDataset(Dataset):
    dataset: List[CUBDNDataItem]
    transforms: Callable
    use_img_tensor: bool

    def __init__(
        self,
        dataset: List[CUBDNDataItem],
        cub_img_dir: str,
        transform: Optional[Callable] = None,
        use_img_tensor: bool = True,
    ):
        self.dataset = dataset
        self.cub_img_dir = cub_img_dir
        self.transform = (
            transform
            if transform
            else transforms.Compose(
                [
                    transforms.CenterCrop(INCEPTION_INPUT_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[2, 2, 2]),
                ]
            )
        )
        self.use_img_tensor = use_img_tensor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx) -> Dict:
        data = self.dataset[idx]
        if self.use_img_tensor:
            img_tensor = self.transform(
                Image.open(f"{self.cub_img_dir}/{data.img_path}").convert("RGB")
            )
        else:
            # If not using img tensor, e.g. train/eval DNF classifier, then the
            # img tensor is just 0, which wouldn't be accessed anyway
            img_tensor = torch.zeros(DUMMY_IMG_TENSOR_SHAPE)

        return asdict(data.to_cub_dataset_item(img_tensor))
