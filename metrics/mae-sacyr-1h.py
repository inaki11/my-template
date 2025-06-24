import torch


def build_metric(config=None, debug=False):
    def mae(y_pred, y_true):
        # this function takes as input two tensors: y_pred and y_true
        if debug:
            print("config inside mae:", config)

        assert (
            y_pred.shape == y_true.shape
        ), "'mae calculation error:' Shapes of y_pred and y_true must match"
        # get dataset.name last char that is 1, 3, or 6
        prediction_horizon = config.prediction_horizon

        if debug:
            print(f"MAE: y_pred shape: {y_pred.shape}, y_true shape: {y_true.shape}")
            print(f"MAE: y_pred: {y_pred[:5]}, y_true: {y_true[:5]}")
        # Compute the Mean Absolute Error
        # filter prediction_horizon columns
        len_y = y_pred.shape[1]
        target_columns = list(range(0, len_y, prediction_horizon))
        print(
            f"MAE-sacyr-1h: len_y: {len_y},   prediction_horizon: {prediction_horizon},   target_columns: {target_columns} "
        )
        y_pred = y_pred[:, target_columns]
        y_true = y_true[:, target_columns]
        assert (
            y_pred.shape == y_true.shape
        ), "'mae calculation error:' Shapes of y_pred and y_true must match after filtering"

        if debug:
            print(
                f"MAE-sacyr-1h: y_pred shape: {y_pred.shape}, y_true shape: {y_true.shape}"
            )
            print(f"MAE-sacyr-1h: y_pred: {y_pred[:5]}, y_true: {y_true[:5]}")

        return float(torch.mean(torch.abs(y_pred - y_true)))

    return mae
