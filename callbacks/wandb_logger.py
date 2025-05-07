import wandb
import json
import os
from .base_callbacks import BaseCallback
from omegaconf import OmegaConf


class WandbLogger(BaseCallback):
    def __init__(
        self,
        project="default_project",
        config=None,
        dir="wandb",
        entity=None,
        fold=None,
    ):
        self.project = project
        self.config = config
        self.dir = dir
        self.entity = entity
        self.fold = fold

    def on_train_begin(self, trainer):
        run_name = (
            f"{trainer.config.experiment_id}_fold{trainer.fold}"
            if hasattr(trainer, "fold") and trainer.fold is not False
            else trainer.config.experiment_id
        )
        print(f"Run name: {run_name}")
        wandb.init(
            project=self.project,
            config=OmegaConf.to_container(trainer.config, resolve=True),
            dir=self.dir,
            entity=self.entity,
            name=run_name,
            group=trainer.config.experiment_id,
            sync_tensorboard=True,
        )
        wandb.watch(trainer.model, log="all", log_freq=100)

    def on_validation_end(self, trainer):
        epoch_info = {key: float(value) for key, value in trainer.epoch_info.items()}
        wandb.log(epoch_info)

    def on_train_end(self, trainer):
        # wandb.finish()
        pass


def build_callback(config):
    return WandbLogger(
        project=config.get("project", "default_project"),
        entity=config.get("entity", None),
    )
