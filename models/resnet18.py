import torch.nn as nn
import torchvision.models as models

def build_model(config):
    model = models.resnet18(weights=config.weights)
    model.fc = nn.Linear(model.fc.in_features, config.num_classes)
    return model
