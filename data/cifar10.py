from torchvision.datasets import CIFAR10
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, config, split="train", transform=None):
        self.dataset = CIFAR10(
            root=config.dataset.root,
            train=(split == "train"),
            download=True,
            transform=transform
        )

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)
