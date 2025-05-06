import torch.optim.lr_scheduler as scheduler

def build_scheduler(config, optimizer):
    """
    StepLR: Decays the learning rate of each parameter group by gamma every step_size epochs.
    """
    return scheduler.StepLR(
        optimizer,
        step_size=config.get("step_size", 10),
        gamma=config.get("gamma", 0.1)
    )