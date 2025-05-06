import os
import hashlib
import json
from .base_callbacks import BaseCallback
from omegaconf import OmegaConf
import torch


class ModelCheckpoint(BaseCallback):
     # monitor = "loss" ya es validation, ya que está harcodeado trainer.val_metrics.get(self.monitor)
    def __init__(self, dirpath, monitor="loss", mode="min"):
        self.dirpath = dirpath
        self.monitor = monitor
        self.mode = mode
        self.best = float("inf") if mode == "min" else -float("inf")

    def on_validation_end(self, trainer):
        current = trainer.val_metrics.get(self.monitor)

        if current is None:
            return

        is_better = current < self.best if self.mode == "min" else current > self.best

        if is_better:
            self.best = current
            exp_id = trainer.config.get("experiment_id", "default_experiment")
            exp_dir = os.path.join(self.dirpath, exp_id)
            os.makedirs(exp_dir, exist_ok=True)

            path = os.path.join(exp_dir, "best.pth")
            # Solo guardo el modelos ya que no me interesa reanudar entrenamientos, y solo guardo el mejor epoch sobre val
            torch.save({
                #"epoch": trainer.epoch,
                "model_state": trainer.model.state_dict(),
                #"optimizer_state": trainer.optimizer.state_dict(),
                #"scheduler_state": trainer.scheduler.state_dict() if trainer.scheduler else None,
            }, path)

            # Guardar la configuración en el directorio de experimentos
            OmegaConf.save(trainer.config, os.path.join(exp_dir, "config.yaml"))

def build_callback(config):
    return ModelCheckpoint(
        dirpath=config.get("dirpath", "checkpoints"),
        monitor=config.get("monitor", "val_loss"),
        mode=config.get("mode", "min"),
    )