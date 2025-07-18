import typing as ty
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class LSTM(nn.Module):
    def __init__(self, input_size, output_size, config):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout,
        )
        self.fc = nn.Linear(config.hidden_size, output_size)

    def forward(self, x):

        return x


def build_model(input_size, output_size, config):
    # Extract model parameters from the config (assumes they are under the "model" key)
    # first assert that all required keys are present
    print("Building LSTM model with config:", config)
    required_keys = [
        "hidden_size",
        "num_layers",
        "dropout",
    ]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required model configuration key: {key}")

    return LSTM(
        input_size=input_size,
        output_size=output_size,
        config=config,
    )
