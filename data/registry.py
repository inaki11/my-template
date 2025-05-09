import importlib


def load_dataset(config):
    dataset_module = importlib.import_module(f"data.{config.dataset.name.lower()}")
    get_dataset = getattr(dataset_module, "get_dataset", None)
    if get_dataset is None:
        raise ValueError(
            f"[Dataset] '{config.dataset.name}' no encontrado. Asegúrate de que el módulo tiene una clase 'CustomDataset'."
        )

    train_ds, val_ds, test_ds = get_dataset(config)

    return train_ds, val_ds, test_ds
