import wandb
from omegaconf import OmegaConf
import wandb


def wandb_init(config):

    wandb.init(
        # reinit=True,
        project=config.wandb.project,
        config=OmegaConf.to_container(config, resolve=True),
        dir=config.wandb.dir,
        entity=config.wandb.entity,
        name=config.experiment_id,
        # sync_tensorboard=True,
    )
