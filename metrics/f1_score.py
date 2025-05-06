import torch
from sklearn.metrics import f1_score

def build_metric(config):
    average = getattr(config, "average", "macro")  # puede ser 'macro', 'micro', 'weighted', etc.

    def f1(y_pred, y_true):
        """
        y_pred: Tensor de forma (N, C) o (N,)
        y_true: Tensor de forma (N,)
        """
        # Si y_pred tiene logits, convertir a clases
        if y_pred.ndim == 2:
            y_pred = torch.argmax(y_pred, dim=1)

        y_pred_np = y_pred.detach().cpu().numpy()
        y_true_np = y_true.detach().cpu().numpy()

        # f1_score(y_true_np, y_pred_np, average=average) to float
        return float(f1_score(y_true_np, y_pred_np, average=average))

    return f1
