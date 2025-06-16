from torch.utils.data import Dataset
import torch
import os


class CustomDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)


def get_dataset(config):

    # Obtén el directorio del script actual para construir la ruta relativa
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "datasets", "sacyr", config.dataset.name + ".pt")

    dataset = torch.load(data_path)

    # Test se asigna al ultimo 15% de los datos
    test_size = int(len(dataset) * 0.15)
    test = dataset[-test_size:]
    train = dataset[:-test_size]

    print(f"Sacyr dataset: {config.dataset.name}")
    print(f"Train size: {len(train)}, Test size: {len(test)}")

    train_ds = CustomDataset(train)
    test_ds = CustomDataset(test)

    return train_ds, test_ds