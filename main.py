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
from utils.logger import setup_logger
from utils.seed import seed_everything
from utils.get_experiment_id import get_experiment_id
from utils.load_checkpoint import load_checkpoint
import torch


def main(config_path):
    config = OmegaConf.load(config_path)
    seed_everything(config.training.seed)
    config.experiment_id = get_experiment_id(config)

    # TO DO: WandB
    logger = setup_logger()
    logger.info("Configuración cargada:")
    logger.info(OmegaConf.to_yaml(config))

    # Configuración cuda                            
    # TO DO: Cambiarlo a una forma mas adecuada, y seleccionar la gpu que se quiera usar
    if config.training.device == "cuda":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        config.training.device = device
    
    # Dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(config)

    # Modelo
    model = get_model(config.model).to(config.training.device)

    # Loss, Optimizer, Scheduler, Callbacks y metrics
    criterion = get_loss(config.loss)
    optimizer= get_optimizer(config.optimizer, model.parameters())
    scheduler = get_scheduler(config.scheduler, optimizer)
    callbacks = get_callbacks(config.callbacks)
    metrics = get_metrics(config.metrics)

    # Entrenador
    trainer = BaseTrainer(model, criterion, optimizer, scheduler, config, logger, callbacks, metrics)
    trainer.train(train_loader, val_loader)

    # cargar el mejor checkpoint sobre validación en el modelo, cuya referencia ya se encuentra en el trainer
    load_checkpoint(model, config)

    # Evaluar métricas en el conjunto de validación
    val_metrics = trainer.run_epoch(val_loader, training=False)
    print("Métricas de validación:")
    print(val_metrics)

    # Evaluar métricas en el conjunto de test
    test_metrics = trainer.run_epoch(test_loader, training=False)
    print("Métricas de test:")
    print(test_metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/example.yaml")
    args = parser.parse_args()
    main(args.config)