from collections import Counter
from enum import Enum
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.transforms import transforms

from common import INCEPTION_INPUT_SIZE, CUBDNDataItem
from dataset import CUBDNDataset
from rule_learner import DNFBasedClassifier


################################################################################
#                              Dataset utils                                   #
################################################################################

FULL_PKL_KEYS = ["full_train_pkl", "full_val_pkl", "full_test_pkl"]
PARTIAL_PKL_KEYS = ["partial_train_pkl", "partial_val_pkl", "partial_test_pkl"]


class DataloaderMode(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


def load_partial_cub_data(
    is_training: bool,
    batch_size: int,
    data_path_dict: Dict[str, str],
    use_img_tensor: bool = True,
) -> Tuple[DataLoader, DataLoader] | DataLoader:
    # Use exisiting pkl files
    # data_path_dict should contain specified pkl file path
    return _load_cub_data(
        is_training=is_training,
        batch_size=batch_size,
        data_path_list=[data_path_dict[k] for k in PARTIAL_PKL_KEYS],
        cub_img_dir=data_path_dict["cub_images_dir"],
        use_img_tensor=use_img_tensor,
    )


def load_full_cub_data(
    is_training: bool,
    batch_size: int,
    data_path_dict: Dict[str, str],
    use_img_tensor: bool = True,
) -> Tuple[DataLoader, DataLoader] | DataLoader:
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
    use_img_tensor: bool = True,
) -> Tuple[DataLoader, DataLoader] | DataLoader:
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

    if is_training:
        train_loader = _get_cub_dataloader(
            dataset=_get_data_from_pkl(data_path_list[0]),
            dataloader_mode=DataloaderMode.TRAIN,
            batch_size=batch_size,
            cub_img_dir=cub_img_dir,
            use_img_tensor=use_img_tensor,
        )
        val_loader = _get_cub_dataloader(
            dataset=_get_data_from_pkl(data_path_list[1]),
            dataloader_mode=DataloaderMode.VAL,
            batch_size=batch_size,
            cub_img_dir=cub_img_dir,
            use_img_tensor=use_img_tensor,
        )
        return train_loader, val_loader
    else:
        test_loader = _get_cub_dataloader(
            dataset=_get_data_from_pkl(data_path_list[2]),
            dataloader_mode=DataloaderMode.TEST,
            batch_size=batch_size,
            cub_img_dir=cub_img_dir,
            use_img_tensor=use_img_tensor,
        )
        return test_loader


def _get_cub_dataloader(
    dataset: List[CUBDNDataItem],
    dataloader_mode: DataloaderMode,
    batch_size: int,
    cub_img_dir: str,
    use_img_tensor: bool = True,
) -> DataLoader:
    """
    Get data loaders of CUB dataset
    """
    if dataloader_mode == DataloaderMode.TEST:
        transform = transforms.Compose(
            [
                transforms.CenterCrop(INCEPTION_INPUT_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[2, 2, 2]),
            ]
        )
    else:
        transform = transforms.Compose(
            [
                transforms.ColorJitter(
                    brightness=32 / 255, saturation=(0.5, 1.5)  # type: ignore
                ),
                transforms.RandomResizedCrop(INCEPTION_INPUT_SIZE),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[2, 2, 2]),
            ]
        )

    cub_dn_dataset = CUBDNDataset(
        dataset, cub_img_dir, transform, use_img_tensor
    )
    num_samples = len(cub_dn_dataset)

    # Sampler
    if dataloader_mode == DataloaderMode.TRAIN:
        get_sample_label = lambda i: cub_dn_dataset.__getitem__(i)["label"]
        label_counter = Counter(
            [get_sample_label(i) for i in range(num_samples)]
        )
        sampler = WeightedRandomSampler(
            weights=[
                1 / label_counter[get_sample_label(i)]
                for i in range(num_samples)
            ],
            num_samples=num_samples,
            replacement=True,
        )
    else:
        sampler = None

    # Othere parameters for dataloader
    drop_last = dataloader_mode == DataloaderMode.TRAIN
    loader_batch_size = (
        num_samples if dataloader_mode == DataloaderMode.VAL else batch_size
    )

    return DataLoader(
        dataset=cub_dn_dataset,
        batch_size=loader_batch_size,
        drop_last=drop_last,
        sampler=sampler,
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
            delta_step = step - self.delta_decay_delay
            new_delta_val = self.initial_delta * (
                self.delta_decay_rate ** (delta_step // self.delta_decay_steps)
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
