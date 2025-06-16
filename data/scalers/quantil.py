import numpy as np
import torch
from sklearn.preprocessing import QuantileTransformer


class QuantileScalerWrapper:
    def __init__(self):
        self.X_scaler = None  # Necesito sabel el numero de samples para inicializarlo
        self.y_scaler = None

    def fit(self, data):
        self.X_scaler = QuantileTransformer(
            output_distribution="normal",
            n_quantiles=max(min(len(data) // 30, 1000), 10),
            subsample=int(1e9),
        )
        self.y_scaler = QuantileTransformer(
            output_distribution="normal",
            n_quantiles=max(min(len(data) // 30, 1000), 10),
            subsample=int(1e9),
        )

        X_list, y_list = [], []
        for X, y in data:
            X_list.append(X.detach().cpu().numpy())
            y_list.append(y.detach().cpu().numpy())
        X_all = np.vstack(X_list)
        y_all = np.vstack(y_list)

        self.X_scaler.fit(X_all)
        self.y_scaler.fit(y_all)

        print("QuantileTransformer fitted.")

    def transform(self, data):
        scaled = []
        X_list, y_list = [], []
        for X, y in data:
            X_list.append(X.detach().cpu().numpy())
            y_list.append(y.detach().cpu().numpy())
        X_all = np.vstack(X_list)
        y_all = np.vstack(y_list)
        X_scaled = self.X_scaler.transform(X_all)
        y_scaled = self.y_scaler.transform(y_all)
        for i in range(len(X_list)):
            X_tensor = torch.tensor(X_scaled[i], dtype=torch.float32)
            y_tensor = torch.tensor(y_scaled[i], dtype=torch.float32)
            scaled.append((X_tensor, y_tensor))

        return scaled

    def inverse_transform(self, y_scaled):
        result = []
        y_scaled_np = np.vstack([y.detach().cpu().numpy() for y in y_scaled])
        y_inv_scaled = self.y_scaler.inverse_transform(y_scaled_np)
        for y in y_inv_scaled:
            result.append(torch.tensor(y, dtype=torch.float32))
        return result


def build_scaler(config):
    return QuantileScalerWrapper()
