import torch
import torch.nn.functional as F


class RandomCrop:
    def __init__(self, size, padding, probability):
        self.size = size
        self.padding = padding

    def __call__(self, img):
        if self.padding > 0:
            # Pad along the height and width dimensions (last two dims)
            img = F.pad(
                img,
                (self.padding, self.padding, self.padding, self.padding),
                mode="reflect",
            )
        # Get height and width assuming img shape is (C, H, W)
        _, h, w = img.shape
        top = torch.randint(0, h - self.size + 1, (1,)).item()
        left = torch.randint(0, w - self.size + 1, (1,)).item()
        return img[:, top : top + self.size, left : left + self.size]


def build_transform(config):
    size = config.get("size", 32)
    padding = config.get("padding", 4)
    probability = config.get("probability", 0.5)
    return RandomCrop(size=size, padding=padding, probability=probability)
