from collections import Counter
import pickle
import random
import sys
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.transforms import transforms

from common import INCEPTION_INPUT_SIZE, CUBDNDataItem
from dataset import CUBDNDataset, ImbalancedDatasetSampler
from rule_learner import DNFBasedClassifier


################################################################################
#                              Dataset utils                                   #
################################################################################

FULL_PKL_KEYS = ["full_train_pkl", "full_val_pkl", "full_test_pkl"]
PARTIAL_PKL_KEYS = ["partial_train_pkl", "partial_val_pkl", "partial_test_pkl"]


def load_partial_cub_data(
    is_training: bool,
    batch_size: int,
    data_path_dict: Dict[str, str],
    selected_classes: Optional[List[int]] = None,
    random_select: Optional[int] = None,
    use_img_tensor: bool = True,
) -> Union[Tuple[DataLoader, DataLoader], DataLoader]:
    if not selected_classes and not random_select:
        # Use exisiting pkl files
        # data_path_dict should contain specified pkl file path
        return _load_cub_data(
            is_training=is_training,
            batch_size=batch_size,
            data_path_list=[data_path_dict[k] for k in PARTIAL_PKL_KEYS],
            cub_img_dir=data_path_dict["cub_images_dir"],
            use_img_tensor=use_img_tensor,
        )

    # TODO: implement selected classes and random select


def load_full_cub_data(
    is_training: bool,
    batch_size: int,
    data_path_dict: Dict[str, str],
    use_img_tensor: bool = True,
) -> Union[Tuple[DataLoader, DataLoader], DataLoader]:
    return _load_cub_data(
        is_training=is_training,
        batch_size=batch_size,
        data_path_list=[data_path_dict[k] for k in FULL_PKL_KEYS],
        cub_img_dir=data_path_dict["cub_images_dir"],
        use_img_tensor=use_img_tensor,
    )


def _load_cub_data(
    is_training: bool,
    batch_size: int,
    data_path_list: List[str],
    cub_img_dir: str,
    selected_classes: Optional[List[int]] = None,
    use_img_tensor: bool = True,
) -> Union[Tuple[DataLoader, DataLoader], DataLoader]:
    """Load partial CUB data according to selected classes

    Args:
        is_training (bool): is training or not.
        batch_size (int): batch size.
        data_path_list (List[str]): list of pickle pile string, HAS TO BE IN THE
        OREDR of train, val and test.
        cub_img_dir (str): CUB dataset images directory.
        selected_classes (Optional[List[int]], optional): selected classes.
        Defaults to None.
        use_img_tensor (bool, optional): whether load images as tensor. If not,
        dummy image tensor are created instead of loading the image from file.
        Defaults to True.

    Returns:
        Union[Tuple[DataLoader, DataLoader], DataLoader]: target dataloader(s)
    """

    def _get_data_from_pkl(data_pkl_path: str) -> List[CUBDNDataItem]:
        import pickle

        with open(data_pkl_path, "rb") as f:
            return pickle.load(f)

    def _get_partial_data(data_pkl_path: str) -> List[CUBDNDataItem]:
        assert selected_classes
        full_data = _get_data_from_pkl(data_pkl_path)
        return [d for d in full_data if d.label in selected_classes]

    data_collect_fn = (
        _get_partial_data if selected_classes else _get_data_from_pkl
    )

    if is_training:
        train_loader = _get_cub_dataloader(
            dataset=data_collect_fn(data_path_list[0]),
            is_training=is_training,
            batch_size=batch_size,
            cub_img_dir=cub_img_dir,
            use_img_tensor=use_img_tensor,
        )
        val_loader = _get_cub_dataloader(
            dataset=data_collect_fn(data_path_list[1]),
            is_training=is_training,
            batch_size=batch_size,
            cub_img_dir=cub_img_dir,
            use_img_tensor=use_img_tensor,
        )
        return train_loader, val_loader
    else:
        test_loader = _get_cub_dataloader(
            dataset=data_collect_fn(data_path_list[2]),
            is_training=is_training,
            batch_size=batch_size,
            cub_img_dir=cub_img_dir,
            use_img_tensor=use_img_tensor,
        )
        return test_loader


def _get_cub_dataloader(
    dataset: List[CUBDNDataset],
    is_training: bool,
    batch_size: int,
    cub_img_dir: str,
    use_img_tensor: bool = True,
) -> DataLoader:
    """
    Get data loaders of CUB dataset
    """
    if is_training:
        transform = transforms.Compose(
            [
                transforms.ColorJitter(
                    brightness=32 / 255, saturation=(0.5, 1.5)
                ),
                transforms.RandomResizedCrop(INCEPTION_INPUT_SIZE),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[2, 2, 2]),
            ]
        )
    else:
        transform = transforms.Compose(
            [
                transforms.CenterCrop(INCEPTION_INPUT_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[2, 2, 2]),
            ]
        )

    drop_last = is_training

    dataset = CUBDNDataset(dataset, cub_img_dir, transform, use_img_tensor)

    num_samples = len(dataset)
    get_sample_label = lambda i: dataset.__getitem__(i)["label"]
    label_counter = Counter([get_sample_label(i) for i in range(num_samples)])
    samples_weight = torch.Tensor(
        [1 / label_counter[get_sample_label(i)] for i in range(num_samples)]
    )

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        drop_last=drop_last,
        sampler=WeightedRandomSampler(
            samples_weight, num_samples, replacement=True
        ),
    )


def get_dnf_classifier_x_and_y(
    data: dict, use_cuda: bool
) -> Tuple[Tensor, Tensor]:
    """
    Get ground truth x and y for DNF Based Classifier
    x: attribute label/attribute score
    y: class label
    """
    x = 2 * data["attr_present_label"] - 1

    raw_label = data["label"]
    y = raw_label - 1

    if use_cuda:
        x = x.to("cuda")
        y = y.to("cuda")
    return x, y


################################################################################
#                                 DNF utils                                    #
################################################################################


class DeltaDelayedExponentialDecayScheduler:
    initial_delta: float
    delta_decay_delay: int
    delta_decay_steps: int
    delta_decay_rate: float

    def __init__(
        self,
        initial_delta: float,
        delta_decay_delay: int,
        delta_decay_steps: int,
        delta_decay_rate: float,
    ):
        self.initial_delta = initial_delta
        self.delta_decay_delay = delta_decay_delay
        self.delta_decay_steps = delta_decay_steps
        self.delta_decay_rate = delta_decay_rate

    def step(self, model: DNFBasedClassifier, step: int) -> float:
        if step < self.delta_decay_delay:
            new_delta_val = self.initial_delta
        else:
            new_delta_val = self.initial_delta * (
                self.delta_decay_rate ** (step // self.delta_decay_steps)
            )
        new_delta_val = 1 if new_delta_val > 1 else new_delta_val
        model.set_delta_val(new_delta_val)
        return new_delta_val


################################################################################
#                                 Model management                             #
################################################################################


def load_pretrained_model_state_dict(
    model: nn.Module, use_cuda: bool, model_pth: str
):
    if use_cuda:
        pretrain_dict = torch.load(model_pth)
        model.load_state_dict(pretrain_dict)
        model.to("cuda")
    else:
        pretrain_dict = torch.load(model_pth, map_location=torch.device("cpu"))
        model.load_state_dict(pretrain_dict)


def freeze_model(model: nn.Module):
    for _, param in model.named_parameters():
        param.requires_grad = False


def gen_las_example(pkl_data_path: str, save_file_path: str) -> int:
    def gen_example_from_data(sample: CUBDNDataItem, file=sys.stdout):
        # Penalty and inclusion set
        print(
            f"#pos(eg_{sample.img_id}@{10}, "
            f"{{class({sample.label - 1})}}, {{",
            file=file,
        )

        # Exclusion set
        exclusion_set = ",\n".join(
            filter(
                lambda j: j != "",
                map(
                    lambda k: f"    class({k})"
                    if k != sample.label - 1
                    else "",
                    range(3),
                ),
            )
        )
        print(exclusion_set, file=file)

        print("}, {", file=file)

        # Context
        for i, attr in enumerate(sample.attr_present_label.int()):
            if attr.item() == 0:
                continue
            print(f"    has_attr_{i}.", file=file)

        print("}).\n", file=file)

    with open(pkl_data_path, "rb") as f:
        cub_train = pickle.load(f)

    with open(save_file_path, "w") as f:
        for sample in cub_train:
            gen_example_from_data(sample, f)
