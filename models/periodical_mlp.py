from rtdl_num_embeddings import PeriodicEmbeddings
import math
import typing as ty
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from torch import Tensor
from typing import Optional


def _named_sequential(*modules) -> nn.Sequential:
    return nn.Sequential(OrderedDict(modules))


# NOTE[DIFF]
# In the paper, for MLP, the width of the first and the last layers is tuned separately
# from the rest of the layers. It turns out that using the same width for all layers
# is not worse, at least not on the datasets used in the paper.
class MLP(nn.Module):
    """The MLP model from Section 3.1 in the paper."""

    def __init__(
        self,
        *,
        d_in: int,
        d_out: Optional[int],
        n_blocks: int,
        d_block: int,
        dropout: float,
    ) -> None:
        """
        Args:
            d_in: the input size.
            d_out: the output size.
            n_blocks: the number of blocks.
            d_block: the block width.
            dropout: the dropout rate.
        """
        if n_blocks <= 0:
            raise ValueError(f"n_blocks must be positive, however: {n_blocks=}")

        super().__init__()
        self.blocks = nn.ModuleList(
            [
                _named_sequential(
                    ("linear", nn.Linear(d_block if i else d_in, d_block)),
                    ("activation", nn.ReLU()),
                    ("dropout", nn.Dropout(dropout)),
                )
                for i in range(n_blocks)
            ]
        )
        self.output = None if d_out is None else nn.Linear(d_block, d_out)

    def forward(self, x: Tensor) -> Tensor:
        for block in self.blocks:
            x = block(x)
        if self.output is not None:
            x = self.output(x)
        return x


# Las siguientes dos clases son creadas por mi enteras para crear modelos con embeddings.
# No distingo entre features numéricas y categóricas
class PeriodicMLP(nn.Module):
    def __init__(self, input_size, output_size, config):
        super(PeriodicMLP, self).__init__()
        self.embedding = PeriodicEmbeddings(
            input_size,
            config.d_embedding,
            n_frequencies=config.n_frequencies,
            frequency_init_scale=config.frequency_init_scale,
            lite=False,
        )
        self.flatten = nn.Flatten()
        self.mlp = MLP(
            d_in=config.d_embedding * input_size,
            d_out=output_size,
            n_blocks=config.num_layers,
            d_block=config.hidden_size,
            dropout=config.dropout,
        )

    def forward(self, x):
        x = self.embedding(x)
        x = self.flatten(x)
        x = self.mlp(x)
        return x


def build_model(input_size, output_size, config):
    # Extract model parameters from the config (assumes they are under the "model" key)
    # first assert that all required keys are present
    print("Building PeriodicMLP model with config:", config)
    required_keys = [
        "num_layers",
        "hidden_size",
        "dropout",
        "d_embedding",
        "n_frequencies",
        "frequency_init_scale",
    ]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required model configuration key: {key}")

    return PeriodicMLP(
        input_size=input_size,
        output_size=output_size,
        config=config,
    )  # Aquí config es un objeto con los parámetros del modelo
