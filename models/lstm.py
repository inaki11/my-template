import typing as ty
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


# En las LSTM al parecer se debe representar la entrada como (batch_size, sequence_length, input_size)
#  Es decir en mi caso si hay 3 espiras y damos 24h de contexto (batch_size, 24, 3)
# Soluciono esta forma en el get_dataset()
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
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])
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
    print(
        f"lstm module:  input_size: {input_size}, output_size: {output_size}, config: {config}"
    )

    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required model configuration key: {key}")

    return LSTM(
        input_size=input_size[0],
        output_size=output_size,
        config=config,
    )
