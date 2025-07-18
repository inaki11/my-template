import torch.nn as nn
import torch.nn.functional as F


class CNN_LSTM(nn.Module):
    """
    Arquitectura híbrida:
    1) Conv1d extrae patrones locales a lo largo de la dimensión temporal
    2) ReLU + Dropout
    3) LSTM modela dependencias de largo alcance
    4) Capa densa final
    ----------
    Forma de entrada esperada: (batch, seq_len, input_size) = (batch, 24, 2)
    """

    def __init__(self, input_size, output_size, config):
        super().__init__()

        # ---------- Bloque CNN ----------
        # Conv1d espera (batch, channels, seq_len) → usamos permute en forward
        self.conv1 = nn.Conv1d(
            in_channels=input_size,  # 2 “canales” (las dos variables)
            out_channels=config.cnn_filters,  # p. ej. 32
            kernel_size=config.kernel_size,  # p. ej. 3
            padding="same",  # mantiene longitud 24
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(config.dropout)

        # ---------- Bloque LSTM ----------
        self.lstm = nn.LSTM(
            input_size=config.cnn_filters,  # lo que salga del Conv1d
            hidden_size=config.hidden_size,  # p. ej. 64
            num_layers=config.num_layers,  # p. ej. 2
            batch_first=True,
            dropout=config.dropout,
        )

        # ---------- Capa de salida ----------
        self.fc = nn.Linear(config.hidden_size, output_size)

    def forward(self, x):
        # x: (batch, 24, 2)  →  (batch, 2, 24)
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.dropout(x)
        # x: (batch, filters, 24)  →  (batch, 24, filters)
        x = x.permute(0, 2, 1)

        # LSTM: devolvemos la última salida temporal
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])
        return x


def build_model(input_size, output_size, config, model_type="lstm"):

    print("Building CNN-LSTM model with config:", config)
    required_keys = [
        "hidden_size",
        "num_layers",
        "dropout",
        "cnn_filters",
        "kernel_size",
    ]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required model configuration key: {key}")
    print(
        f"cnn_lstm module:  input_size: {input_size}, output_size: {output_size}, config: {config}"
    )

    return CNN_LSTM(
        input_size=input_size[0],
        output_size=output_size,
        config=config,
    )
