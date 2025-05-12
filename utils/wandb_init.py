import wandb
from omegaconf import OmegaConf
import wandb


def wandb_init(config):
    # if config.kfold is a key:
    if hasattr(config, "kfold") and config.kfold is not None:
        fold = getattr(config.kfold, "fold", 1)
        run_name = f"{config.experiment_id}_fold{fold}"
    else:
        run_name = config.experiment_id

    print(f"Run name: {run_name}")
    wandb.init(
        reinit=True,
        project=config.wandb.project,
        config=OmegaConf.to_container(config, resolve=True),
        dir=config.wandb.dir,
        entity=config.wandb.entity,
        name=run_name,
        group=config.experiment_id,
        sync_tensorboard=True,
    )
