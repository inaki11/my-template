import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

class StandardScalerWrapper:
    def __init__(self):
        self.X_scaler = StandardScaler()
        self.y_scaler = StandardScaler()

    def fit(self, data):
        X_list, y_list = [], []
        for X, y in data:
            X_list.append(X.detach().cpu().numpy())
            y_list.append(y.detach().cpu().numpy())
        X_all = np.vstack(X_list)
        y_all = np.vstack(y_list)

        self.X_scaler.fit(X_all)
        self.y_scaler.fit(y_all)

        print("StandardScaler fitted:")
        print(f"X mean: {self.X_scaler.mean_}, X var: {self.X_scaler.var_}")
        print(f"y mean: {self.y_scaler.mean_}, y var: {self.y_scaler.var_}")

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
        # y_scaled is a list of tensors
        # first we convert them to numpy matrix
        y_scaled_np = np.vstack([y.detach().cpu().numpy() for y in y_scaled])
        y_inv_scaled = self.y_scaler.inverse_transform(y_scaled_np)
        for y in y_inv_scaled:
            result.append(torch.tensor(y, dtype=torch.float32))
        return result
    
def build_scaler(config):
    return StandardScalerWrapper()
    