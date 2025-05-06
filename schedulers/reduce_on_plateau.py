import torch.optim.lr_scheduler as scheduler

def build_scheduler(config, optimizer):
    """
    ReduceLROnPlateau: Reduce learning rate when a metric has stopped improving.
    This is useful for reducing the learning rate when the model is not converging.
    """
    class ReduceLROnPlateauWrapper:
        def __init__(self, config, optimizer):
            self.scheduler = scheduler.ReduceLROnPlateau(
                optimizer,
                mode=config.get("mode", "min"),
                factor=config.get("factor", 0.1),
                patience=config.get("patience", 10),
                threshold=config.get("threshold", 1e-4),
                threshold_mode=config.get("threshold_mode", "rel"),
                cooldown=config.get("cooldown", 0),
                min_lr=config.get("min_lr", 0),
                eps=config.get("eps", 1e-8),
            )

        def step(self, trainer):
            # Use the validation loss metric to step the scheduler
            val_loss = trainer.val_metrics.get("loss")
            if val_loss is not None:
                self.scheduler.step(val_loss)
            else:
                trainer.logger.warning("Validation loss not found. Scheduler step skipped.")

    return ReduceLROnPlateauWrapper(config, optimizer)