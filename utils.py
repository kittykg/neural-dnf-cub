from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from common import INCEPTION_INPUT_SIZE, CUBDNDataItem
from dataset import CUBDNDataset
from rule_learner import DNFBasedClassifier


################################################################################
#                              Dataset utils                                   #
################################################################################


def load_partial_cub_data(
    is_training: bool,
    batch_size: int,
    data_path_dict: Dict[str, str],
    selected_classes: Optional[List[int]] = None,
    use_img_tensor: bool = True,
) -> Union[Tuple[DataLoader, DataLoader], DataLoader]:

    return


def load_full_cub_data(
    is_training: bool,
    batch_size: int,
    data_path_dict: Dict[str, str],
    use_img_tensor: bool = True,
) -> Union[Tuple[DataLoader, DataLoader], DataLoader]:
    if is_training:
        train_loader = _get_cub_dataloader(
            dataset=_get_data(data_path_dict["train_pkl"]),
            is_training=is_training,
            batch_size=batch_size,
            cub_img_dir=data_path_dict["cub_images_dir"],
            use_img_tensor=use_img_tensor,
        )
        val_loader = _get_cub_dataloader(
            dataset=_get_data(data_path_dict["val_pkl"]),
            is_training=is_training,
            batch_size=batch_size,
            cub_img_dir=data_path_dict["cub_images_dir"],
            use_img_tensor=use_img_tensor,
        )
        return train_loader, val_loader
    else:
        test_loader = _get_cub_dataloader(
            dataset=_get_data(data_path_dict["test_pkl"]),
            is_training=is_training,
            batch_size=batch_size,
            cub_img_dir=data_path_dict["cub_images_dir"],
            use_img_tensor=use_img_tensor,
        )
        return test_loader


def _get_data(data_pth_path: str) -> List[CUBDNDataItem]:
    import pickle

    with open(data_pth_path, "rb") as f:
        return pickle.load(f)


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
    shuffle = is_training

    return DataLoader(
        CUBDNDataset(dataset, cub_img_dir, transform, use_img_tensor),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
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
