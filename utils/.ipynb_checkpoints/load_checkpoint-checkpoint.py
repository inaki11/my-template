import torch
from omegaconf import OmegaConf

def load_checkpoint(model, config, fold):
    checkpoint = torch.load(f"checkpoints/{config.experiment_id}/best_{fold}.pth", map_location=config.training.device)
    if "model_state" in checkpoint:
        state_dict = checkpoint["model_state"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)