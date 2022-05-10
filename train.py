import logging
from typing import Callable, Dict, Iterable, OrderedDict

import matplotlib.pyplot as plt
from omegaconf import DictConfig
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
import wandb

from analysis import MetricValueMeter, MultiClassAccuracyMeter
from rule_learner import DNFBasedClassifier
from utils import (
    load_full_cub_data,
    get_dnf_classifier_x_and_y,
    DeltaDelayedExponentialDecayScheduler,
)


log = logging.getLogger()

FULL_PKL_KEYS = ["full_train_pkl", "full_val_pkl", "full_test_pkl"]
PARTIAL_PKL_KEYS = ["partial_train_pkl", "partial_val_pkl", "partial_test_pkl"]

loss_func_map: Dict[str, Callable[[Tensor, Tensor], Tensor]] = {
    "bce": lambda y_hat, y: torch.mean(
        torch.sum(F.binary_cross_entropy(y_hat, y, reduction="none"), 1)
    ),
    "ce": nn.CrossEntropyLoss(),
}


class DnfClassifierTrainer:
    # Data loaders
    train_loader: DataLoader
    val_loader: DataLoader

    # Training parameters
    use_cuda: bool
    experiment_name: str
    optimiser_key: str
    optimiser_fn: Callable[[Iterable], Optimizer]
    scheduler_fn: Callable
    loss_func_key: str
    criterion: Callable[[Tensor, Tensor], Tensor]
    epochs: int
    reg_lambda: float

    # Training meters, should be setup by _meters_setup()
    train_loss_meter: MetricValueMeter
    train_performance_score_meter: MultiClassAccuracyMeter
    val_performance_score_meter: MultiClassAccuracyMeter

    # Delta decay scheduler
    delta_decay_scheduler: DeltaDelayedExponentialDecayScheduler
    delta_one_counter: int = -1

    # Configs
    cfg: DictConfig
    model_train_cfg: DictConfig

    def __init__(self, model_name: str, cfg: DictConfig) -> None:
        # Configs
        self.cfg = cfg
        self.model_train_cfg = cfg["training"][model_name]

        # Training parameters
        self.use_cuda = (
            cfg["training"]["use_cuda"] and torch.cuda.is_available()
        )
        self.experiment_name = cfg["training"]["experiment_name"]

        # Data loaders
        env_cfg = cfg["environment"]
        use_partial_cub = cfg["training"]["use_partial_cub"]
        partial_cub_cfg = cfg["training"]["partial_cub"]

        if use_partial_cub and "selected_classes" in partial_cub_cfg:
            # Use selected classes
            pass
        elif use_partial_cub and 'random_select' in partial_cub_cfg:
            # Randomly select a number of classes, based on 'random_select'
            pass
        elif use_partial_cub:
            # Use existing pkl files
            pass
        else:
            for k in FULL_PKL_KEYS:
                assert k in env_cfg
            data_path_dict = {}
            for k1, k2 in zip(
                ["train_pkl", "val_pkl", "test_pkl"], FULL_PKL_KEYS
            ):
                data_path_dict[k1] = env_cfg[k2]
            data_path_dict["cub_images_dir"] = env_cfg["cub_images_dir"]

        self.train_loader, self.val_loader = load_full_cub_data(
            is_training=True,
            batch_size=self.model_train_cfg["batch_size"],
            data_path_dict=data_path_dict,
            use_img_tensor=False,
        )

        # Optimiser
        lr = self.model_train_cfg["optimiser_lr"]
        weight_decay = self.model_train_cfg["optimiser_weight_decay"]
        self.optimiser_key = self.model_train_cfg["optimiser"]
        if self.optimiser_key == "sgd":
            self.optimiser_fn = lambda params: torch.optim.SGD(
                params, lr=lr, momentum=0.9, weight_decay=weight_decay
            )
        else:
            self.optimiser_fn = lambda params: torch.optim.Adam(
                params, lr=lr, weight_decay=weight_decay
            )

        # Scheduler
        scheduler_step = self.model_train_cfg["scheduler_step"]
        self.scheduler_fn = lambda optimiser: torch.optim.lr_scheduler.StepLR(
            optimiser, step_size=scheduler_step, gamma=0.1
        )

        # Loss function
        self.loss_func_key = self.model_train_cfg["loss_func"]
        self.criterion = loss_func_map[self.loss_func_key]

        # Other training parameters
        self.epochs = self.model_train_cfg["epochs"]
        self.reg_lambda = self.model_train_cfg["reg_lambda"]

        # Training meters
        self.train_loss_meter = MetricValueMeter(
            f"train_{self.optimiser_key}_loss"
        )
        self.train_performance_score_meter = MultiClassAccuracyMeter()
        self.val_performance_score_meter = MultiClassAccuracyMeter()

        self.delta_decay_scheduler = DeltaDelayedExponentialDecayScheduler(
            initial_delta=self.model_train_cfg["initial_delta"],
            delta_decay_delay=self.model_train_cfg["delta_decay_delay"],
            delta_decay_steps=self.model_train_cfg["delta_decay_steps"],
            delta_decay_rate=self.model_train_cfg["delta_decay_rate"],
        )

    def train(self, model: DNFBasedClassifier) -> OrderedDict:
        seed = torch.get_rng_state()[0].item()
        log.info(f"{self.experiment_name} starts, seed: {seed}")

        if self.use_cuda:
            model.to("cuda")

        optimiser = self.optimiser_fn(model.parameters())
        scheduler = self.scheduler_fn(optimiser)

        for epoch in range(self.epochs):
            # 1. Training
            self._epoch_train(epoch, model, optimiser)

            # 2. Evaluate performance on val
            self._epoch_val(epoch, model)

            # 3. Let scheduler update optimiser at end of epoch
            scheduler.step()

        return model.state_dict()

    def _epoch_train(
        self, epoch: int, model: DNFBasedClassifier, optimiser: Optimizer
    ) -> None:
        epoch_loss_meter = MetricValueMeter("epoch_loss_meter")
        epoch_perf_score_meter = MultiClassAccuracyMeter()

        model.train()

        for i, data in enumerate(self.train_loader):
            optimiser.zero_grad()

            x, y = get_dnf_classifier_x_and_y(data, self.use_cuda)
            y_hat = model(x)

            loss = self._loss_calculation(y_hat, y, model.parameters())

            # Update meters
            epoch_loss_meter.update(loss.item())
            self.train_loss_meter.update(loss.item())

            epoch_perf_score_meter.update(y_hat, y)
            self.train_performance_score_meter.update(y_hat, y)

        # Update delta value
        new_delta_val = self.delta_decay_scheduler.step(model, epoch)

        if new_delta_val == 1.0:
            # The first time where new_delta_val becomes 1, the network isn't
            # train with delta being 1 for that epoch. So delta_one_counter
            # starts with -1, and when new_delta_val is first time being 1,
            # the delta_one_counter becomes 0.
            self.delta_one_counter += 1

        # Log average performance for train
        avg_loss = epoch_loss_meter.get_average()
        avg_perf = epoch_perf_score_meter.get_average()
        log.info(
            "[%3d] Train  Delta: %.3f  avg loss: %.3f  avg perf: %.3f"
            % (epoch + 1, new_delta_val, avg_loss, avg_perf)
        )

        # Generate weight histogram
        sd = model.state_dict()
        conj_w = sd["dnf.conjunctions.weights"].flatten().detach().cpu().numpy()
        disj_w = sd["dnf.disjunctions.weights"].flatten().detach().cpu().numpy()

        f1 = plt.figure(figsize=(20, 15))
        plt.title("Conjunction weight distribution")
        arr = plt.hist(conj_w, bins=20)
        for i in range(20):
            plt.text(arr[1][i], arr[0][i], str(int(arr[0][i])))

        f2 = plt.figure(figsize=(20, 15))
        plt.title("Disjunction weight distribution")
        arr = plt.hist(disj_w, bins=20)
        for i in range(20):
            plt.text(arr[1][i], arr[0][i], str(int(arr[0][i])))

        # WandB logging
        wandb.log(
            {
                "train/epoch": epoch + 1,
                "delta": new_delta_val,
                "train/loss": avg_loss,
                "train/accuracy": avg_perf,
                "conj_w_hist": f1,
                "disj_w_hist": f2,
            }
        )

    def _epoch_val(self, epoch: int, model: DNFBasedClassifier) -> float:
        epoch_val_loss_meter = MetricValueMeter("epoch_val_loss_meter")
        epoch_val_perf_score_meter = MultiClassAccuracyMeter()

        model.eval()

        for i, data in enumerate(self.val_loader):
            with torch.no_grad():
                # Get model output and compute loss
                x, y = get_dnf_classifier_x_and_y(data, self.use_cuda)
                y_hat = model(x)
                loss = self._loss_calculation(y_hat, y, model.parameters())

            # Update meters
            epoch_val_loss_meter.update(loss)
            epoch_val_perf_score_meter.update(y_hat, y)
            self.val_performance_score_meter.update(y_hat, y)

        avg_loss = epoch_val_loss_meter.get_average()
        avg_perf = epoch_val_perf_score_meter.get_average()
        log.info(
            "[%3d] Val                  avg loss: %.3f  avg perf: %.3f"
            % (epoch + 1, avg_loss, avg_perf)
        )

        wandb.log(
            {
                "val/epoch": epoch + 1,
                "val/loss": avg_loss,
                "val/accuracy": avg_perf,
            }
        )

        return avg_perf

    def _loss_calculation(
        self,
        y_hat: Tensor,
        y: Tensor,
        parameters: Iterable[nn.parameter.Parameter],
    ) -> Tensor:
        loss = self.criterion(y_hat, y)

        if self.delta_one_counter >= 10:
            # Extra regularisation when delta has been 1 more than for 10.
            # Pushes weights towards 0, -6 or 6.
            def weight_regulariser(w: Tensor):
                return torch.abs(w * (6 - torch.abs(w))).sum()

            reg = self.reg_lambda * sum(
                [weight_regulariser(p.data) for p in parameters]
            )
            loss += reg

        return loss
