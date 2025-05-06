import torch.optim as optim

def build_optimizer(config, params):
    return optim.Adam(params, lr=config.lr, weight_decay=config.weight_decay)
