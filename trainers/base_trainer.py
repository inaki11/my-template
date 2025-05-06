import torch
from tqdm import tqdm


class BaseTrainer:
    def __init__(self, model, criterion, optimizer, scheduler, config, logger, callbacks=None, metrics=None):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.logger = logger
        self.device = config.training.device
        self.callbacks = callbacks if callbacks is not None else []
        self.metrics = metrics if metrics is not None else []
        self.train_metrics = {}
        self.val_metrics = {}

    def train(self, train_loader, val_loader):
        for cb in self.callbacks:
            cb.on_train_begin(self)

        num_epochs = self.config.training.epochs

        for epoch in range(num_epochs):
            self.epoch = epoch

            self.train_metrics = self.run_epoch(train_loader, training=True)
            for cb in self.callbacks:
                cb.on_epoch_end(self)

            self.val_metrics = self.run_epoch(val_loader, training=False)
            for cb in self.callbacks:
                cb.on_validation_end(self)

            lr = self.optimizer.param_groups[0]["lr"]
            # Log metrics
            self.logger.info({
                "Epoch": epoch + 1,
                "Train_loss": f"{self.train_metrics['loss']:.4f}",
                "Val_loss": f"{self.val_metrics['loss']:.4f}",
                "LR": f"{lr:.2e}",
                **{f"val_{metric.__name__}": f"{self.val_metrics[metric.__name__]:.4f}" for metric in self.metrics}
            })

            if self.scheduler:
                self.scheduler.step(self)

        for cb in self.callbacks:
            cb.on_train_end(self)


    def run_epoch(self, loader, training=True):
        self.model.train() if training else self.model.eval()
        total_loss = 0.0
        total_batches = 0
        all_outputs = []

        mode = "Train" if training else "Val"
        pbar = tqdm(loader, desc=f"{mode} Epoch {self.epoch+1}", leave=False, ncols=100)

        with torch.set_grad_enabled(training):
            for inputs, targets in pbar:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                if training:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                total_loss += loss.item()
                total_batches += 1

                pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / total_batches

        return {"loss": avg_loss, **{metric.__name__: metric(outputs, targets) for metric in self.metrics}}
