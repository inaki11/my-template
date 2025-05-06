# data/__init__.py
from .registry import load_dataset
from .preprocessing import get_preprocess_transforms
from data.augmentations import get_augmentation_transforms
from torch.utils.data import DataLoader

def prepare_datasets(config):
    train_ds, val_ds, test_ds = load_dataset(config)
    return train_ds, val_ds, test_ds

def add_transforms(train_ds, val_ds, test_ds, config):
    """
    Las transformaciones en PyTorch no se aplican directamente al cargar el dataset, 
    sino que se ejecutan cuando accedes a un ítem, es decir, dentro del __getitem__() del Dataset.

    Para que esto funcione, debe guardarse en el atributo 'transform' del dataset la transformacion o collate de transformaciones.
    
    """
    train_tfms, val_tfms, test_tfms = None, None, None

    # 1. Preprocess (compartido)
    if config.get("preprocessing"):
        preprocess = get_preprocess_transforms(config.preprocessing)
        # añado las transformaciones de preprocesado a ambos datasets
        train_tfms =  preprocess
        val_tfms = preprocess
        test_tfms = preprocess

    # 2. Augmentations (solo train)
    if config.get("augmentations"):
        train_aug = get_augmentation_transforms(config.augmentations, split="train")
        # se añaden las transformaciones de augmentations solo al dataset de train
        train_tfms = train_tfms + train_aug if config.get("preprocessing") else train_tfms

    # Al cargar el dataset estamos envolviendo el dataset original en un CustomDataset, 
    # que tiene a su vez un atributo interno dataset "dataset", por ejemplo el CIFAR10
    # Hay que acceder al atributo transform dentro del dataset original para que se aplique la transformacion
    train_ds.dataset.transform = train_tfms
    val_ds.dataset.transform = val_tfms
    test_ds.dataset.transform = test_tfms

    return train_ds, val_ds, test_ds

def get_dataloaders(config):
    train_ds, val_ds, test_ds = prepare_datasets(config)
    train_ds, val_ds, test_ds = add_transforms(train_ds, val_ds, test_ds, config)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader