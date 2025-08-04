import torch


def build_metric(config=None, debug=False):
    def mae(y_pred, y_true):
        # input: 2 lists of tensors
        if isinstance(y_pred, list):
            y_pred = torch.stack(y_pred)
        if isinstance(y_true, list):
            y_true = torch.stack(y_true)
        # Compute the Mean Absolute Error per column
        return torch.mean(torch.abs(y_pred - y_true), dim=0)

    return mae
