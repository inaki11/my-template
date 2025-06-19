import torch
from tqdm import tqdm


class BaseTrainer:
    def __init__(
        self,
        model,
        criterion,
        optimizer,
        scheduler,
        config,
        logger,
        fold,
        callbacks=None,
        metrics=None,
    ):
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
        self.fold = fold
        self.stop = False  # Flag for early stopping

    def train(self, train_loader, val_loader):
        for cb in self.callbacks:
            cb.on_train_begin(self)

        num_epochs = self.config.training.epochs

        for epoch in range(num_epochs):
            self.epoch = epoch

            self.train_metrics = self.run_epoch(train_loader, mode="Train")
            for cb in self.callbacks:
                cb.on_epoch_end(self)

            self.val_metrics = self.run_epoch(val_loader, mode="Val")
            lr = self.optimizer.param_groups[0]["lr"]
            # Log metrics
            self.epoch_info = {
                "Train_loss": f"{self.train_metrics['Train_loss']:.4f}",
                **self.val_metrics,
                "lr": f"{lr:.2e}",
            }
            self.logger.info(self.epoch_info)

            for cb in self.callbacks:
                cb.on_validation_end(self)

            if self.scheduler:
                self.scheduler.step(self)

            # Stop if early stopping is triggered
            if self.stop:
                self.logger.info("Early stopping triggered. Stopping training.")
                break

        for cb in self.callbacks:
            cb.on_train_end(self)

    def run_epoch(self, loader, mode="Train", return_preds=False):
        training = mode == "Train"
        self.model.train() if training else self.model.eval()

        total_loss = 0.0
        total_batches = 0

        all_inputs = []
        all_outputs = []
        all_targets = []

        pbar = tqdm(
            loader, desc=f"{mode} Epoch {self.epoch+1}", ncols=100, dynamic_ncols=True
        )

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

                if return_preds:
                    all_inputs.append(inputs.cpu())
                    all_outputs.append(outputs.cpu())
                    all_targets.append(targets.cpu())

                for cb in self.callbacks:
                    cb.on_batch_end(self)

        avg_loss = total_loss / total_batches

        # Solo calcula las métricas sobre el último batch si no se están guardando todos los outputs
        if not return_preds:
            metric_values = {
                f"{mode}_{metric.__name__}": metric(outputs, targets)
                for metric in self.metrics
            }
        else:
            cat_outputs = torch.cat(all_outputs)
            cat_targets = torch.cat(all_targets)
            metric_values = {
                f"{mode}_{metric.__name__}": metric(cat_outputs, cat_targets)
                for metric in self.metrics
            }

        metrics = {
            f"{mode}_loss": avg_loss,
            **metric_values,
        }

        if return_preds:
            return (
                metrics,
                torch.cat(all_inputs),
                torch.cat(all_outputs),
                torch.cat(all_targets),
            )
        else:
            return metrics
