import torch.nn as nn

def build_loss(config):
    return nn.CrossEntropyLoss()
