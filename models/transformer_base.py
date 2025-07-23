import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, : x.size(1)]
        return x


class TransformerTimeSeries(nn.Module):
    """
    Transformer base para series temporales univariadas o multivariadas.
    """

    def __init__(self, input_size, output_size, config):
        super().__init__()

        self.embedding_dim = config.embed_dim_per_head * config.nhead

        # Proyecto cada paso temporal (input_size) a embedding_dim
        self.input_proj = nn.Linear(input_size, self.embedding_dim)

        self.pos_encoder = PositionalEncoding(d_model=self.embedding_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embedding_dim,
            nhead=config.nhead,
            dim_feedforward=config.ff_dim,
            dropout=config.dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers,
        )

        # Capa de salida: usamos la representación del último paso temporal
        self.fc_out = nn.Linear(self.embedding_dim, output_size)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        x = self.input_proj(x)  # (batch, seq_len, embedding_di)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)  # (batch, seq_len, embedding_dim)
        x = x[:, -1, :]  # último paso temporal
        x = self.fc_out(x)  # (batch, output_size)
        return x


def build_model(input_size, output_size, config):
    print(f"Building Transformer_Base model with config:", config)

    required_keys = ["embed_dim_per_head", "nhead", "ff_dim", "num_layers", "dropout"]

    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required model configuration key: {key}")
    return TransformerTimeSeries(
        input_size=input_size[0],
        output_size=output_size,
        config=config,
    )
