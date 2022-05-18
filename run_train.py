import os
import random

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch
import wandb


from rule_learner import DNFClassifier, DNFClassifierEO, DNFBasedClassifier
from train import DnfClassifierTrainer


@hydra.main(config_path="conf", config_name="config")
def run_experiment(cfg: DictConfig) -> None:
    experiment_name = cfg["training"]["experiment_name"]
    model_name = cfg["training"]["model_name"]

    # Set up wandb
    run = wandb.init(
        project="cub-3",
        entity="kittykg",
        config=OmegaConf.to_container(cfg["training"][model_name]),
    )

    # Set random seed
    random_seed = cfg["training"]["random_seed"]
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)

    # Set up model
    model_class = DNFClassifier if model_name == "dnf" else DNFClassifierEO
    base_cfg = OmegaConf.to_container(cfg["model"]["base_dnf"])
    model = model_class(**base_cfg)
    model.set_delta_val(cfg["training"][model_name]["initial_delta"])

    torch.autograd.set_detect_anomaly(True)

    trainer = DnfClassifierTrainer(model_name, cfg)
    state_dict = trainer.train(model)

    # Save model
    torch.save(state_dict, f"{experiment_name}_{random_seed}.pth")
    model_artifact = wandb.Artifact(
        f"{experiment_name}",
        type="model",
        description=f"{experiment_name} model",
        metadata=dict(wandb.config),
    )
    model_artifact.add_file(f"{experiment_name}_{random_seed}.pth")
    wandb.save(f"{experiment_name}_{random_seed}.pth")
    run.log_artifact(model_artifact)


if __name__ == "__main__":
    # os.environ["WANDB_MODE"] = "disabled"
    run_experiment()
