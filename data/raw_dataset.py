from torch.utils.data import Dataset

class RawDataset(Dataset):
    def __init__(self, config, transform=None):
        self.data = ...  # cargar los datos crudos
        self.targets = ...  # cargar etiquetas
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx], self.targets[idx]
        if self.transform:
            x = self.transform(x)
        return x, y
