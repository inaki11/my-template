import torch


def build_metric(config=None, debug=False):
    def mae(y_pred, y_true):
        # this function takes as input two tensors: y_pred and y_true
        # both tensors have the same shape, (N, C) or (N,)
        # if is (N, C) is a multi-regression problem
        # if is (N,) is a single regression problem

        # Ensure y_pred and y_true are the same shape
        assert (
            y_pred.shape == y_true.shape
        ), "'mae calculation error:' Shapes of y_pred and y_true must match"

        if debug:
            print(f"MAE: y_pred shape: {y_pred.shape}, y_true shape: {y_true.shape}")
            print(f"MAE: y_pred: {y_pred[:5]}, y_true: {y_true[:5]}")
        # Compute the Mean Absolute Error
        return float(torch.mean(torch.abs(y_pred - y_true)))

    return mae
