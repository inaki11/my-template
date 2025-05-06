import torch.optim.lr_scheduler as scheduler

def build_scheduler(config, optimizer):
    """
    CosineAnnealingWarmRestarts: Cosine LR decay con reinicios periódicos.
    Útil para escapar de mínimos locales o mejorar generalización.
    """
    class CosineAnnealingWarmRestartsWrapper:
        def __init__(self, config, optimizer):
            self.scheduler = scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=config.get("t_0", 10),                # Número de epochs hasta el primer reinicio
                T_mult=config.get("t_mult", 1),          # Factor multiplicativo entre reinicios
                eta_min=config.get("eta_min", 0),        # LR mínimo
                last_epoch=config.get("last_epoch", -1),
            )

        def step(self, **kwargs):
            self.scheduler.step()

    return CosineAnnealingWarmRestartsWrapper(config, optimizer)
