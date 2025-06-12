# data/__init__.py
from .registry import load_train_test_set
from .preprocessing import get_preprocess_transforms
from .scalers import get_scaler
from data.augmentations import get_augmentation_transforms
from torch.utils.data import DataLoader, Subset, Dataset

from sklearn.model_selection import train_test_split, KFold

def get_validation_set(train_ds, config):
    """
    Divide el dattaser en base a la configuración. Si el yaml tiene 'kfold' se divide en k folds, 
    si no, se divide en train y val, reservando un 20% del train.
    Si la configuración del dataset tiene el modo 'classification', se dividen train y val de forma estratificada.

    Ejemplo yaml:
    dataset:
        name: CIFAR10
        dataset_module: CIFAR10
        root: ./data/datasets
        num_workers: 4
        mode: reggresion  # classification
        kfold:
            num_folds: 5


    Args:
        train_data: lista de tuplas (input, target) que son tensores de PyTorch.
    """
    # Si no hay kfold, se divide en train y val
    if not config.dataset.get("kfold"):

        # Si el dataset es de clasificación, se divide de forma estratificada
        if config.dataset.mode == "classification":
            targets = None  # TO DO
            train_ds, val_ds = train_test_split(
                train_ds,
                test_size=0.2,
                stratify=targets, # TO DO (targets no es nada)
                random_state=config.training.seed,
            )
        else:
            train_ds, val_ds = train_test_split(
                train_ds, test_size=0.2, random_state=config.training.seed
            )
    else:
        if config.dataset.mode == "classification":
            # TO DO:  si es classify estos folds deben ser estratificados
            print("TO DO:  si es classify estos folds deben ser estratificados")
            pass
        else:
            # Si hay kfold, se divide en k folds, 
            kfold = config.dataset.kfold.num_folds
            # Si el parametro shuffle es True, se barajan los datos de train antes de dividirlos usando una semilla
            kf = KFold(n_splits=kfold, shuffle=config.dataset.kfold.get("shuffle"), random_state=config.training.seed if config.dataset.kfold.get("shuffle") else None)
            train_ds_list, val_ds_list = [], []
            for train_index, val_index in kf.split(train_ds):
                train_subset = Subset(train_ds, train_index)
                val_subset = Subset(train_ds, val_index)
                train_ds_list.append(train_subset)
                val_ds_list.append(val_subset)
    return train_ds_list, val_ds_list

class CustomDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples   = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        # Transformaciones CPU ligeras
        if self.transform is not None:
            x = self.transform(x)
        return x, y
         
def get_dataset(train_ds, val_ds, test_ds, config):

    return (
        CustomDataset(train_ds, transform=None),
        CustomDataset(val_ds, transform=None),
        CustomDataset(test_ds, transform=None)
    )

def apply_scaler(train_set, test_set, config, debug=False):
    scaler = get_scaler(config.scaler)
    if scaler is None:
        print("No se ha especificado un escalador. No se aplicará ninguna transformación.")
        return train_set
    scaler.fit(train_set)
    train_set_scaled = scaler.transform(train_set)
    test_set_scaled = scaler.transform(test_set)
    if debug:
        print("antes de aplicar el escalador:")
        print("train_set:", train_set[:5])
        print("test_set:", test_set[:5])
        print("después de aplicar el escalador:")
        print("train_set_scaled:", train_set_scaled[:5])
        print("test_set_scaled:", test_set_scaled[:5])
    return train_set_scaled, test_set_scaled, scaler

    


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
        train_tfms = preprocess
        val_tfms = preprocess
        test_tfms = preprocess

    # 2. Augmentations (solo train)
    if config.get("augmentations"):
        train_aug = get_augmentation_transforms(config.augmentations, split="train")
        # se añaden las transformaciones de augmentations solo al dataset de train
        train_tfms = (
            train_tfms + train_aug if config.get("preprocessing") else train_tfms
        )

    # Al cargar el dataset estamos envolviendo el dataset original en un CustomDataset,
    # que tiene a su vez un atributo interno dataset "dataset", por ejemplo el CIFAR10
    # Hay que acceder al atributo transform dentro del dataset original para que se aplique la transformacion
    train_ds.transform = train_tfms
    if val_ds is not None:
        # Si no hay dataset de validacion, no se aplica la transformacion
        val_ds.transform = val_tfms
    if test_ds is not None:
        # Si no hay dataset de test, no se aplica la transformacion
        test_ds.transform = test_tfms

    return train_ds, val_ds, test_ds


def get_dataloaders(train_ds, val_ds, test_ds, config):
    train_ds, val_ds, test_ds = add_transforms(train_ds, val_ds, test_ds, config)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        prefetch_factor=config.training.prefech_factor,
        pin_memory=True,
    )
    if val_ds is None:
        val_loader = None
    else:
        val_loader = DataLoader(
            val_ds,
            batch_size=config.training.batch_size,
            shuffle=False,
            num_workers=config.training.num_workers,
            prefetch_factor=config.training.prefech_factor,
            pin_memory=True,
        )
    if test_ds is None:
        test_loader = None
    else:
        test_loader = DataLoader(
            test_ds,
            batch_size=config.training.batch_size,
            shuffle=False,
            num_workers=config.training.num_workers,
            prefetch_factor=config.training.prefech_factor,
            pin_memory=True,
        )

    return train_loader, val_loader, test_loader
