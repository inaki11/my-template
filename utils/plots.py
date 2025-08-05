import matplotlib.pyplot as plt
import wandb
from sklearn.metrics import mean_absolute_error
import torch


def plot_inverse_transformed_outputs_and_mae(outputs, targets, fold):
    """
    Plots some days of the inverse transformed outputs and targets, and calculates the Mean Absolute Error (MAE).
    Logs the plot to Weights & Biases (wandb) with the current fold.

    Args:
        outputs: The model's predictions after inverse transformation.
        targets: The true values after inverse transformation.
        fold: The current fold number for logging purposes.
    """
    print("Logging inverse transformed outputs plots and MAE...")
    mae = mean_absolute_error(targets, outputs)

    multi_regression = outputs[0].shape[0] > 1

    if multi_regression:
        predictions_length = outputs[0].shape[0]
        indices = [i * predictions_length for i in range(1, 7)]
        targets_to_plot = [targets[idx] for idx in indices]
        outputs_to_plot = [outputs[idx] for idx in indices]
        # join all the tensors in the list
        targets_to_plot = torch.cat(targets_to_plot, dim=0)
        outputs_to_plot = torch.cat(outputs_to_plot, dim=0)

    else:
        targets_to_plot = targets[0:144]
        outputs_to_plot = outputs[0:144]

    plt.figure(figsize=(10, 5))
    plt.plot(targets_to_plot, label="Targets", color="green")
    plt.plot(outputs_to_plot, label="Outputs", color="orange")
    plt.title(f"Fold {fold} - Outputs vs Targets 1st week\nMAE: {mae:.4f}")
    plt.ylabel("traffic flow")
    plt.legend()
    plt.grid()

    # Loggear la imagen a wandb
    wandb.log(
        {
            f"Outputs_vs_Targets_1st_week_Fold_{fold}": wandb.Image(
                plt.gcf(), caption=f"MAE: {mae:.4f}"
            )
        }
    )
    # Cierra la figura para liberar memoria
    plt.close()


def plot_average_mae_per_step(mae_per_step_folds):
    # hacemos la media de mae_per_step_folds
    values = torch.stack(mae_per_step_folds).mean(dim=0)
    print(f"MAE per step: {values}")
    labels = list(range(len(values)))  # Simple integer labels
    table = wandb.Table(
        data=[[i, val] for i, val in zip(labels, values)], columns=["Step", "MAE"]
    )
    wandb.log({"Bar Plot": wandb.plot.bar(table, "Step", "MAE", title="MAE per step")})
