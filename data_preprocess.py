import argparse
import pickle
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from common import (
    CUBRawDataItem,
    CUBDNDataItem,
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


################################################################################
#                       Run below for preprocessing                            #
################################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-rtrain", type=str, help="Raw train pkl path")
    parser.add_argument("-rval", type=str, help="Raw val pkl path")
    parser.add_argument("-rtest", type=str, help="Raw test pkl path")
    parser.add_argument(
        "-nc", type=int, help="Number of classes after filtering"
    )
    parser.add_argument("-t", type=int, help="Min class threshold")
    parser.add_argument(
        "-od", type=str, help="Output directory (no trailing '/')"
    )
    args = parser.parse_args()

    num_classes = args.nc
    threshold = args.t
    target_classes = list(range(1, num_classes + 1))
    out_dir = args.od

    print("Start pre-process")

    # Train dataset
    with open(args.rtrain, "rb") as f:
        train_raw = pickle.load(f)
    train_dataset, median, mask = majority_voting(
        dataset=[d for d in train_raw if d.label in target_classes],
        total_number_classes=num_classes,
        min_class_threshold=threshold,
    )
    with open(f"{out_dir}/train.pkl", "wb") as f:
        pickle.dump(train_dataset, f)
    print("Processed train dataset stored")

    # Val dataset
    with open(args.rval, "rb") as f:
        val_raw = pickle.load(f)
    val_dataset, _, _ = majority_voting(
        dataset=[d for d in val_raw if d.label in target_classes],
        total_number_classes=num_classes,
        min_class_threshold=threshold,
        precomputed_mask=mask,
        precomputed_class_attr_median=median,
    )
    with open(f"{out_dir}/val.pkl", "wb") as f:
        pickle.dump(val_dataset, f)
    print("Processed val dataset stored")

    # Test dataset
    with open(args.rtest, "rb") as f:
        test_raw = pickle.load(f)
    test_dataset, _, _ = majority_voting(
        dataset=[d for d in test_raw if d.label in target_classes],
        total_number_classes=num_classes,
        min_class_threshold=threshold,
        precomputed_mask=mask,
        precomputed_class_attr_median=median,
    )
    with open(f"{out_dir}/test.pkl", "wb") as f:
        pickle.dump(test_dataset, f)
    print("Processed test dataset stored")

    # Store mask and median
    with open(f"{out_dir}/mask.pkl", "wb") as f:
        pickle.dump(mask, f)
    with open(f"{out_dir}/median.pkl", "wb") as f:
        pickle.dump(median, f)
    print("Median and mask stored")

    print()
    print("---------------Summary---------------")
    print(f"Number of classes:         {num_classes}")
    print(f"Min class threshold:       {threshold}")
    print(f"Number of attributes used: {len(mask)}")
    print("-------------------------------------")
