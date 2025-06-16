import torch.nn as nn

def build_loss(config):
    # Para regresión múltiple se recomienda MSELoss
    return nn.MSELoss()