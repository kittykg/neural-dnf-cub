from typing import List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from common import (
    CUBRawDataItem,
    CUBDNDataItem,
    CUB_NUM_CLASSES,
    CUB_NUM_ALL_ATTRIBUTES,
)


def majority_voting(
    dataset: List[CUBRawDataItem],
    total_number_classes: int,
    min_class_threshold: int,
    precomputed_mask: Optional[np.ndarray] = None,
    precomputed_class_attr_median: Optional[Tensor] = None,
) -> Tuple[List[CUBDNDataItem], Tensor, np.ndarray]:
    """
    The majority voting process described in the CBM model. It computes the
    median label of an attribute in a class, if the attribute is present in at
    least n classes consistently.
    Can take precomputed mask and median.
    Return the new dataset, the median, and the mask (selected attributes)
    """
    if (
        precomputed_mask is not None
        and precomputed_class_attr_median is not None
    ):
        mask = precomputed_mask
        class_attr_median = precomputed_class_attr_median
    else:
        class_attr_count = np.zeros(
            (total_number_classes, CUB_NUM_ALL_ATTRIBUTES, 2)
        )
        for d in dataset:
            for a in d.attributes:
                if a.is_present == 0 and a.certainty == 1:
                    # ignore the not present because not visible
                    continue
                class_attr_count[d.label - 1][a.attr_id - 1][a.is_present] += 1

        # For each class, for each attribute, which label is less frequent
        # (present or not present)
        class_attr_min_label = np.argmin(class_attr_count, axis=2)
        # For each class, for each attribute, which label is more frequent
        # (present or not present)
        class_attr_max_label = np.argmax(class_attr_count, axis=2)

        # Find where most and least frequent are equal, set most frequent to 1
        equal_count = np.where(class_attr_min_label == class_attr_max_label)
        class_attr_max_label[equal_count] = 1

        if precomputed_mask is not None:
            mask = precomputed_mask
        else:
            # Count number of times each attribute is mostly present for a class
            attr_class_count = np.sum(class_attr_max_label, axis=0)

            # Select the attributes that are present most of the time,
            # on a class level, in at least `min_class_threshold` classes
            mask = np.where(attr_class_count >= min_class_threshold)[0]

        class_attr_median = torch.from_numpy(
            class_attr_max_label[:, mask]
        ).float()

    new_dataset = []
    for d in dataset:
        attr_present_label = class_attr_median[[d.label - 1], :].squeeze()
        # Set the attr_certainty to level 4 (100% certain), this shouldn't be
        # required in training/testing
        attr_certainty = torch.ones(attr_present_label.shape) * 4
        new_dataset.append(
            CUBDNDataItem(
                img_id=d.img_id,
                img_path=d.img_path,
                label=d.label,
                attr_present_label=attr_present_label,
                attr_certainty=attr_certainty,
            )
        )

    return new_dataset, class_attr_median, mask
