import torch.optim as optim

def build_optimizer(config, params):
    return optim.SGD(params, lr=config.lr, momentum=config.get("momentum", 0.9))
