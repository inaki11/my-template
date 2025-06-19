import wandb
from .base_callbacks import BaseCallback


class WandbLogger(BaseCallback):

    def on_validation_end(self, trainer):
        epoch_info = {key: float(value) for key, value in trainer.epoch_info.items()}
        wandb.log(epoch_info)

    def on_train_end(self, trainer):
        # wandb.finish()
        pass


def build_callback(config):
    return WandbLogger()
