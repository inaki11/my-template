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

    logger = setup_logger()
    logger.info("Configuración cargada:")
    logger.info(config)

    # print visible GPUs
    print("--------------   GPU  --------------------")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Utilizando device: {device}")

    # init wandb, Una run por cada conjunto de folds, ya que sweeps no me deja hacer varios inits en la misma run.
    wandb_init(config)

    train_set, test_set = load_train_test_set(config)
    input_size, output_size = train_set[0][0].shape[0], train_set[0][1].shape[0]
    logger.info(f"Input size: {input_size}, Output size: {output_size}")

    train_set_scaled, test_set_scaled, scaler = apply_scaler(
        train_set, test_set, config, debug=True
    )  # Aplica el escalador si es necesario
    train_splits, val_splits = get_validation_set(
        train_set_scaled, config
    )  # Devuelve una lista. Si es sin kfold solo un elemento en train y otro en val. Si es k-fold devuelve los k sets.

    folds_val_metrics, folds_test_metrics = {}, {}
    for fold, (train_fold, val_fold) in enumerate(zip(train_splits, val_splits)):
        print(
            f"-------------------------------\n Entrenando fold {fold + 1}/{len(train_splits)} \n -------------------------------"
        )

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
            model,
            criterion,
            optimizer,
            scheduler,
            config,
            logger,
            fold,
            callbacks,
            metrics,
        )
        trainer.train(train_loader, val_loader)

        load_checkpoint(model, config, fold)
        val_metrics, inputs, outputs, targets = trainer.run_epoch(
            val_loader, mode="Val", return_preds=True
        )
        print("Métricas de validación:")
        print(val_metrics)

        # Acumulamos metricas de validación
        for key, value in val_metrics.items():
            if key not in folds_val_metrics:
                folds_val_metrics[key] = []
            folds_val_metrics[key].append(value)

        test_metrics = trainer.run_epoch(test_loader, mode="Test")
        print("Métricas de test:")
        print(test_metrics)

        # Acumulamos metricas de Test
        for key, value in test_metrics.items():
            if key not in folds_test_metrics:
                folds_test_metrics[key] = []
            folds_test_metrics[key].append(value)

    # Final del k-fold loggueamos las loss medias para optimizar la búsqueda de hiperparámetros
    if getattr(config.dataset, "kfold", False):
        logger.info("K-Fold Training Complete, calculating mean metrics.")

        print(folds_val_metrics)
        print(folds_test_metrics)
        # Log the mean metrics for k-fold
        for key in val_metrics.keys():
            metric_mean = sum(val_metrics[key]) / len(val_metrics[key])
            logger.info(f"Mean_{key}: {metric_mean:.4f}")
            wandb.log({f"Mean_{key}": metric_mean})
        for key in test_metrics.keys():
            metric_mean = sum(test_metrics[key]) / len(test_metrics[key])
            logger.info(f"Mean_{key}: {metric_mean:.4f}")
            wandb.log({f"Mean_{key}": metric_mean})

    else:
        logger.info("Training Complete, logging final losses.")
        for key, value in val_metrics.items():
            wandb.log({f"{key}": value[0]})
        for key, value in test_metrics.items():
            wandb.log({f"{key}": value[0]})

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    args, _ = parser.parse_known_args()

    main(args.config)
