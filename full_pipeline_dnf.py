from datetime import datetime
import os
import pickle
import random
import requests
import traceback
from typing import Dict

import hydra
from hydra.core.hydra_config import HydraConfig
import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch
import wandb


from rule_learner import DNFBasedClassifier, DNFClassifier, DNFClassifierEO
from train import DnfClassifierTrainer
from dnf_post_train import (
    VanillaDNFPostTrainingProcessor,
    DNFEOPostTrainingProcessor,
)


def post_to_discord_webhook(webhook_url: str, message: str) -> None:
    requests.post(webhook_url, json={"content": message})


def convert_result_dict_to_discord_message(
    experiment_name: str,
    random_seed: int,
    is_eo_based: bool,
    rd: Dict[str, float],
) -> str:
    nodename = os.uname().nodename
    dt = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    s = (
        f"[{dt}]\nExperiment {experiment_name} (seed {random_seed}) "
        f"on Machine {nodename} has finished!\n"
    )

    if is_eo_based:
        s += f"Results (on test set):"
        s += f"""```
        After train     Acc    Jacc
            DNF-EO      {rd['after_train_acc']}  {rd['after_train_jacc']}
            DNF         {rd['pd_after_train_acc']}  {rd['pd_after_train_jacc']}
        After prune     Acc    Jacc
            DNF                {rd['after_prune_test']}
        After tune      Acc    Jacc
            DNF-EO      {rd['after_tune_acc']}  {rd['after_tune_jacc']}
            DNF         {rd['pd_after_tune_acc']}  {rd['pd_after_tune_jacc']}
        After thresh    Acc    Jacc
            DNF                {rd['after_threshold_test_jacc']}
        Rule extract    Acc    Jacc
            DNF         {rd['rule_acc']}  {rd['rule_jacc']}```
        """
    else:
        s += f"Results (on test set):"
        s += f"""```
                            Acc    Jacc
            After train     {rd['after_train_acc']}  {rd['after_train_jacc']}
            After prune            {rd['after_prune']}
            After tune      {rd['after_tune_acc']}  {rd['after_tune_jacc']}
            After thresh           {rd['after_threshold_test_jacc']}
            Rule extract    {rd['rule_acc']}  {rd['rule_jacc']}```
        """
    return s


def run_train(
    cfg: DictConfig, experiment_name: str, random_seed: int
) -> DNFBasedClassifier:
    model_name = cfg["training"]["model_name"]

    # Set up wandb
    run = wandb.init(
        project="cub-3",
        entity="kittykg",
        config=OmegaConf.to_container(cfg["training"][model_name]),  # type: ignore
        dir=HydraConfig.get().run.dir,
    )

    # Set up model
    model_class = (
        DNFClassifier if model_name == "dnf_vanilla" else DNFClassifierEO
    )
    base_cfg = OmegaConf.to_container(cfg["model"]["base_dnf"])
    model = model_class(**base_cfg)  # type: ignore
    model.set_delta_val(cfg["training"][model_name]["initial_delta"])

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
    run.log_artifact(model_artifact)  # type: ignore

    return model


def run_post_training_processing(
    model_name: str, cfg: DictConfig, model: DNFBasedClassifier
) -> Dict[str, float]:
    post_train_processor = (
        DNFEOPostTrainingProcessor
        if isinstance(model, DNFClassifierEO)
        else VanillaDNFPostTrainingProcessor
    )(model_name, cfg)

    result_dict = post_train_processor.post_processing(model)  # type: ignore
    return result_dict


@hydra.main(version_base=None, config_path="conf", config_name="config")
def run_pipeline(cfg: DictConfig) -> None:
    # Config parameters
    random_seed = cfg["training"]["random_seed"]
    experiment_name = cfg["training"]["experiment_name"]
    model_name = cfg["training"]["model_name"]
    is_eo_based = model_name == "dnf_eo"
    webhook_url = cfg["webhook"]["discord_webhook_url"]

    # Set random seed
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)

    # Train and post-training process
    try:
        model = run_train(cfg, experiment_name, random_seed)
        result_dict = run_post_training_processing(model_name, cfg, model)
        with open(
            f"{experiment_name}_full_pipeline_result_dict.pkl", "wb"
        ) as f:
            pickle.dump(result_dict, f)
        webhook_msg = convert_result_dict_to_discord_message(
            experiment_name, random_seed, is_eo_based, result_dict
        )
    except BaseException as e:
        nodename = os.uname().nodename
        dt = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        webhook_msg = (
            f"[{dt}]\nExperiment {experiment_name} (seed {random_seed}) "
            f"on Machine {nodename} got an error!! Check that out!!"
        )
        print(traceback.format_exc())
    finally:
        post_to_discord_webhook(webhook_url, webhook_msg)  # type: ignore


if __name__ == "__main__":
    os.environ["WANDB_MODE"] = "disabled"
    run_pipeline()
