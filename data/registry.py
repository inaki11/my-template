import importlib

def load_dataset(config):
    dataset_module = importlib.import_module(f"data.{config.dataset.name.lower()}")
    dataset_class = getattr(dataset_module, "CustomDataset", None)
    if dataset_class is None:
        raise ValueError(f"[Dataset] '{config.dataset.name}' no encontrado. Asegúrate de que el módulo tiene una clase 'CustomDataset'.")
    
    train_ds = dataset_class(config, split="train")
    val_ds = dataset_class(config, split="val")
    test_ds = dataset_class(config, split="test")

    return train_ds, val_ds, test_ds