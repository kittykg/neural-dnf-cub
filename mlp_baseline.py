from datetime import datetime
import logging
import os
import random
import requests
import traceback
from typing import Dict

import hydra
import numpy as np
from omegaconf import DictConfig
import torch
from torch import nn, Tensor

from analysis import (
    MetricValueMeter,
    MultiClassAccuracyMeter,
    JaccardScoreMeter,
)
from utils import load_partial_cub_data, get_dnf_classifier_x_and_y

#    Statistics
# Data     IN  OUT
# CUB-3   (34,  3)
# CUB-10  (41,  10)
# CUB-15  (40,  15)
# CUB-20  (48,  20)
# CUB-25  (50,  25)
# CUB-50  (61,  50)
# CUB-100 (82,  100)
# CUB-200 (112, 200)

# MLP baseline experiment hyperparameter
# MLP architecture: 2 layers, IN x (3 X OUT) x OUT
ATTR_IN = 112
LABEL_OUT = 200
INTERMEDIATE_LAYER = LABEL_OUT * 3
EXPERIMENT_NAME = f"mlp_cub_{LABEL_OUT}"

USE_CUDA: bool = True
BATCH_SIZE: int = 32
LR: float = 0.001
WEIGHT_DECAY: float = 0.00004
EPOCHS: int = 200
RANDOM_SEED: int = 73

log = logging.getLogger()


class MLP(nn.Module):
    def __init__(self) -> None:
        super(MLP, self).__init__()
        self.l1 = nn.Linear(ATTR_IN, INTERMEDIATE_LAYER)
        self.a1 = nn.Tanh()
        self.l2 = nn.Linear(INTERMEDIATE_LAYER, LABEL_OUT)

    def forward(self, input: Tensor) -> Tensor:
        y = self.l1(input)
        y = self.a1(y)
        y = self.l2(y)
        return y


def train(model: MLP, data_path_dict: Dict[str, str]) -> None:
    if USE_CUDA:
        model.to("cuda")

    train_loader, val_loader = load_partial_cub_data(
        is_training=True,
        batch_size=BATCH_SIZE,
        data_path_dict=data_path_dict,
        use_img_tensor=False,
    )

    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(
        model.parameters(), LR, weight_decay=WEIGHT_DECAY
    )

    for e in range(EPOCHS):
        # TRAIN
        epoch_loss_meter = MetricValueMeter("epoch_loss_meter")
        epoch_perf_score_meter = MultiClassAccuracyMeter()
        model.train()
        for data in train_loader:
            optimiser.zero_grad()
            x, y = get_dnf_classifier_x_and_y(data, USE_CUDA)
            y_hat = model(x)
            loss = criterion(y_hat, y)
            loss.backward()
            optimiser.step()
            epoch_loss_meter.update(loss.item())
            epoch_perf_score_meter.update(y_hat, y)
        avg_loss = epoch_loss_meter.get_average()
        avg_perf = epoch_perf_score_meter.get_average()
        log.info(
            "[%3d] Train   avg loss: %.3f  avg perf: %.3f"
            % (e + 1, avg_loss, avg_perf)
        )

        # VAL
        epoch_val_loss_meter = MetricValueMeter("epoch_val_loss_meter")
        epoch_val_perf_score_meter = MultiClassAccuracyMeter()
        model.eval()
        for data in val_loader:
            with torch.no_grad():
                x, y = get_dnf_classifier_x_and_y(data, USE_CUDA)
                y_hat = model(x)
                loss = criterion(y_hat, y)
                epoch_val_loss_meter.update(loss.item())
                epoch_val_perf_score_meter.update(y_hat, y)
        avg_loss = epoch_val_loss_meter.get_average()
        avg_perf = epoch_val_perf_score_meter.get_average()
        log.info(
            "[%3d] Val     avg loss: %.3f  avg perf: %.3f"
            % (e + 1, avg_loss, avg_perf)
        )


def eval(model: MLP, data_path_dict: Dict[str, str]) -> Dict[str, float]:
    model.eval()
    test_loader = load_partial_cub_data(
        is_training=False,
        batch_size=BATCH_SIZE,
        data_path_dict=data_path_dict,
        use_img_tensor=False,
    )
    jaccard_meter = JaccardScoreMeter()
    accuracy_meter = MultiClassAccuracyMeter()
    for data in test_loader:
        with torch.no_grad():
            x, y = get_dnf_classifier_x_and_y(data, USE_CUDA)  # type: ignore
            y_hat = model(x)
            jaccard_meter.update((torch.tanh(y_hat) > 0).long(), y)
            accuracy_meter.update(y_hat, y)
    overall_jacc = jaccard_meter.get_average()
    overall_acc = accuracy_meter.get_average()
    log.info(
        f"Overall Test   avg acc: {overall_acc:.3f}  avg jacc: {overall_jacc:.3f}"
    )
    return {"test_acc": overall_acc, "test_jacc": overall_jacc}


def _run_mlp_baseline_helper(
    data_path_dict: Dict[str, str]
) -> Dict[str, float]:
    model = MLP()
    train(model, data_path_dict)
    torch.save(model.state_dict(), "mlp_baseline.pth")
    return eval(model, data_path_dict)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def run_mlp_baseline(cfg: DictConfig) -> None:
    # Set random seed
    torch.manual_seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # Use hydra config for getting webhook and data pkl path
    data_path_dict = {
        "partial_train_pkl": cfg["training"]["partial_cub"][
            "partial_train_pkl"
        ],
        "partial_val_pkl": cfg["training"]["partial_cub"]["partial_val_pkl"],
        "partial_test_pkl": cfg["training"]["partial_cub"]["partial_test_pkl"],
        "cub_images_dir": "",  # dummy as we don't load the image
    }
    webhook_url = cfg["webhook"]["discord_webhook_url"]
    nodename = os.uname().nodename

    def post_to_discord_webhook(webhook_url: str, message: str) -> None:
        requests.post(webhook_url, json={"content": message})

    try:
        res_dict = _run_mlp_baseline_helper(data_path_dict)
        dt = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        webhook_msg = (
            f"[{dt}]\n"
            f"Experiment MLP Baseline {EXPERIMENT_NAME} (seed {RANDOM_SEED}) "
            f"on Machine {nodename} FINISHED!!\n"
            f"Result:\n"
            f"Test acc: {res_dict['test_acc']:.3f}\n"
            f"Test jacc: {res_dict['test_jacc']:.3f}"
        )
    except BaseException as e:
        dt = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        webhook_msg = (
            f"[{dt}]\n"
            f"Experiment MLP Baseline {EXPERIMENT_NAME} (seed {RANDOM_SEED}) "
            f"on Machine {nodename} got an error!! Check that out!!"
        )
        print(traceback.format_exc())
    finally:
        post_to_discord_webhook(webhook_url, webhook_msg)  # type: ignore


if __name__ == "__main__":
    run_mlp_baseline()
