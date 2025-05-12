import torch
import random


class RandomHorizontalFlip:
    def __init__(self, probability=0.5):
        self.probability = probability

    def __call__(self, img):
        if random.random() < self.probability:
            # using torch.flip to flip the image horizontally
            return img.flip(dims=[-1]).contiguous()
        return img


def build_transform(config):
    probability = config.get("probability", 0.5)
    return RandomHorizontalFlip(probability=probability)
