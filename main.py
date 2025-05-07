from omegaconf import OmegaConf
import argparse
from models import get_model
from data import get_dataloaders
from optimizers import get_optimizer
from losses import get_loss
from schedulers import get_scheduler
from metrics import get_metrics
from callbacks import get_callbacks
from trainers.base_trainer import BaseTrainer
from loggers import setup_logger, get_output_logger
from utils.loggers import setup_logger
from utils.seed import seed_everything
from utils.get_experiment_id import get_experiment_id
from utils.load_checkpoint import load_checkpoint
from utils.wandb_login import wandb_login
from utils.filter_wrong_predictions import filter_wrong_predictions
import torch
import wandb


def main(config_path):
    config = OmegaConf.load(config_path)
    seed_everything(config.training.seed)
    config.experiment_id = get_experiment_id(config)

    wandb_login()
    logger = setup_logger()
    logger.info("Configuración cargada:")
    logger.info(config)

    if config.training.device == "cuda":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        config.training.device = device

    train_loader, val_loader, test_loader = get_dataloaders(config)
    model = get_model(config.model).to(config.training.device)
    criterion = get_loss(config.loss)
    optimizer = get_optimizer(config.optimizer, model.parameters())
    scheduler = get_scheduler(config.scheduler, optimizer)
    callbacks = get_callbacks(config.callbacks)
    metrics = get_metrics(config.metrics)

    trainer = BaseTrainer(
        model, criterion, optimizer, scheduler, config, logger, callbacks, metrics
    )
    trainer.train(train_loader, val_loader)

    load_checkpoint(model, config)
    val_metrics, inputs, outputs, targets = trainer.run_epoch(
        val_loader, mode="Val", return_preds=True
    )
    print("Métricas de validación:")
    print(val_metrics)

    label_names = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]
    inputs, outputs, targets = filter_wrong_predictions(inputs, outputs, targets)
    output_logger = get_output_logger(config.output_logger)
    output_logger(inputs, outputs, targets, label_names)

    test_metrics = trainer.run_epoch(test_loader, mode="Test")
    print("Métricas de test:")
    print(test_metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/example.yaml")
    args = parser.parse_args()
    main(args.config)
