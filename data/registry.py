import importlib


def load_train_test_set(config):
    dataset_module = importlib.import_module(f"data.{config.dataset.dataset_module.lower()}")
    get_dataset = getattr(dataset_module, "get_dataset", None)
    if get_dataset is None:
        raise ValueError(
            f"[Dataset] '{config.dataset.dataset_module}' no encontrado. Asegúrate de que el módulo tiene una clase 'CustomDataset'."
        )

    train_set, test_set = get_dataset(config)

    return train_set, test_set
