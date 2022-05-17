from dataclasses import dataclass
from typing import List

from torch import Tensor

# Constants
INCEPTION_INPUT_SIZE: int = 299
CUB_NUM_CLASSES: int = 200
CUB_NUM_ALL_ATTRIBUTES: int = 312


# Data structures
@dataclass
class Attribute:
    attr_id: int
    is_present: int
    certainty: int


@dataclass
class CUBDatasetItem:
    """
    Used for dataloader
    """

    img_id: int
    img_path: str
    img_tensor: Tensor
    label: int
    attr_present_label: Tensor
    attr_certainty: Tensor


@dataclass
class CUBRawDataItem:
    img_id: int
    img_path: str
    label: int
    attributes: List[Attribute]


@dataclass
class CUBDNDataItem:
    """
    Used for storing/loading pkl files
    """

    img_id: int
    img_path: str
    label: int
    attr_present_label: Tensor
    attr_certainty: Tensor

    def to_cub_dataset_item(self, img_tensor: Tensor) -> CUBDatasetItem:
        return CUBDatasetItem(
            img_id=self.img_id,
            img_path=self.img_path,
            img_tensor=img_tensor,
            label=self.label,
            attr_present_label=self.attr_present_label,
            attr_certainty=self.attr_certainty,
        )
