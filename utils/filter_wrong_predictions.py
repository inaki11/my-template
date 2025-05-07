def filter_wrong_predictions(inputs, outputs, labels):
    """
    Filter wrong predictions from the outputs and labels.
    Args:
        outputs (torch.Tensor): Model outputs.
        labels (torch.Tensor): True labels.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Filtered outputs and labels.
    """
    preds = outputs.argmax(axis=1)
    mask = preds != labels
    return inputs[mask], outputs[mask], labels[mask]
