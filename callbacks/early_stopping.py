from .base_callbacks import BaseCallback


class early_stopping(BaseCallback):
    def __init__(self, patience=5, min_delta=0.001, monitor="Val_loss"):
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = None
        self.wait = 0

    def on_validation_end(self, trainer):
        current_score = trainer.epoch_info.get(self.monitor)

        if current_score is None:
            return

        if self.best_score is None:
            self.best_score = current_score
            return

        if current_score < self.best_score - self.min_delta:
            self.best_score = current_score
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                print(
                    f"Early stopping: triggered after {self.wait} epochs without '{self.monitor}' improvement."
                )
                trainer.stop = True


def build_callback(config):
    return early_stopping(
        patience=config.get("patience", 5),
        min_delta=config.get("min_delta", 0.001),
        monitor=config.get("monitor", "Val_loss"),
    )
