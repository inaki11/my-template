from torchvision.datasets import CIFAR10
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


def get_dataset(kwargs):

    # Obtén el directorio del script actual para construir la ruta relativa
    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(base_dir, "datasets", "challenge", "train.pt")
    test_path = os.path.join(base_dir, "datasets", "challenge", "test_no_labels.pt")

    train = torch.load(train_path)
    test = torch.load(test_path)

    print(train)
    print(type(train))

    train_ds = CustomDataset(train)
    test_ds = CustomDataset(test)

    # create ds from train and test

    return train_ds, None, test_ds
