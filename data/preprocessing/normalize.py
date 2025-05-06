import numpy as np

class Normalize:
    def __init__(self, mean=0.0, std=1.0):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        return (sample - self.mean) / self.std

def build_transform(config):
    return Normalize(mean=config.get("mean", 0.0), std=config.get("std", 1.0))
