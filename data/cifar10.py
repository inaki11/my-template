from torchvision.datasets import CIFAR10
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, config, split="train", transform=None):
        self.dataset = CIFAR10(
            root=config.dataset.root,
            train=(split == "train"),
            download=True,
            transform=transform,
        )

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)


def get_dataset(config):
    train_ds = CustomDataset(config, split="train")
    val_ds = CustomDataset(config, split="val")
    test_ds = CustomDataset(config, split="test")

    return train_ds, val_ds, test_ds
