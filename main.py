from omegaconf import OmegaConf
import argparse
from models import get_model
from data import apply_scaler, get_validation_set, get_dataset, get_dataloaders
from data.registry import load_train_test_set
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
from utils.wandb_init import wandb_init
import torch
import wandb
import os


def main(config_path):
    # OmegaConf carga la config base
    config = OmegaConf.load(config_path)
    # init vacio para poder usar el sweep
    wandb_login()
    wandb.init()

    # sweep_cfg es la configuración del wandb sweep
    sweep_cfg = OmegaConf.from_dotlist([f"{k}={v}" for k, v in wandb.config.items()])
    print("Configuración del sweep:")
    print(sweep_cfg)
    # merge la configuración base con la del sweep
    config = OmegaConf.merge(config, sweep_cfg)

    # Configuración de la semilla
    seed_everything(config.training.seed)
    # creación del id de experimento
    config.experiment_id = get_experiment_id(config)
    # init con con reinicio y nombre de run basado en hash de la configuración sweep
    wandb_init(config)

    logger = setup_logger()
    logger.info("Configuración cargada:")
    logger.info(config)

    if config.training.device == "cuda":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        config.training.device = device

    print(f"Utilizando device: {device}")

    train_set, test_set = load_train_test_set(config)
    input_size, output_size = train_set[0][0].shape[0], train_set[0][1].shape[0]
    logger.info(f"Input size: {input_size}, Output size: {output_size}")

    train_set_scaled, test_set_scaled, scaler = apply_scaler(
        train_set, test_set, config, debug=True
    )  # Aplica el escalador si es necesario
    train_splits, val_splits = get_validation_set(
        train_set_scaled, config
    )  # Devuelve una lista. Si es sin kfold solo un elemento en train y otro en val. Si es k-fold devuelve los k sets.

    val_losses, test_losses = [], []
    for i, (train_fold, val_fold) in enumerate(zip(train_splits, val_splits)):
        print(f"Entrenando fold {i + 1}/{len(train_splits)}")

        train_ds, val_ds, test_ds = get_dataset(
            train_fold, val_fold, test_set_scaled, config
        )
        train_loader, val_loader, test_loader = get_dataloaders(
            train_ds, val_ds, test_ds, config
        )

        model = get_model(input_size, output_size, config.model).to(
            config.training.device
        )
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
        # Acumulamos Val Loss
        val_losses.append(val_metrics["Val_loss"])

        # inputs, outputs, targets = filter_wrong_predictions(inputs, outputs, targets)

        # if config.output_logger:
        #    output_logger = get_output_logger(config.output_logger)
        #    output_logger(inputs, outputs, targets, label_names)

        test_metrics = trainer.run_epoch(test_loader, mode="Test")
        print("Métricas de test:")
        print(test_metrics)
        # Acumulamos Test Loss
        test_losses.append(test_metrics["Test_loss"])

    # Final del k-fold loggueamos las loss medias para optimizar la búsqueda de hiperparámetros
    if getattr(config.dataset, "kfold", False):
        logger.info("K-Fold Training Complete, calculating mean losses.")
        val_loss_mean = sum(val_losses) / len(val_losses)
        test_loss_mean = sum(test_losses) / len(test_losses)
        logger.info(f"Val Loss Mean: {val_loss_mean}")
        logger.info(f"Test Loss Mean: {test_loss_mean}")
        wandb.log({"Val_loss_mean": val_loss_mean, "Test_loss_mean": test_loss_mean})
    else:
        logger.info("Training Complete, logging final losses.")
        wandb.log({"Val_loss_mean": val_losses[0], "Test_loss_mean": test_losses[0]})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/example.yaml")
    args, _ = parser.parse_known_args()

    main(args.config)
