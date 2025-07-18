import matplotlib.pyplot as plt
import wandb
from sklearn.metrics import mean_absolute_error


def plot_inverse_transformed_outputs_and_mae(outputs, targets, fold):
    """
    Plots the inverse transformed outputs and targets, and calculates the Mean Absolute Error (MAE).
    Logs the plot to Weights & Biases (wandb) with the current fold.

    Args:
        outputs: The model's predictions after inverse transformation.
        targets: The true values after inverse transformation.
        fold: The current fold number for logging purposes.
    """
    print("Logging inverse transformed outputs plots and MAE...")
    mae = mean_absolute_error(targets, outputs)

    plt.figure(figsize=(10, 5))
    plt.plot(targets, label="Targets", color="green")
    plt.plot(outputs, label="Outputs", color="orange")
    plt.title(f"Fold {fold} - Outputs vs Targets\nMAE: {mae:.4f}")
    plt.ylabel("traffic flow")
    plt.legend()
    plt.grid()

    # Loggear la imagen a wandb
    wandb.log(
        {
            f"Outputs_vs_Targets_Fold_{fold}": wandb.Image(
                plt.gcf(), caption=f"MAE: {mae:.4f}"
            )
        }
    )
    # Cierra la figura para liberar memoria
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(targets[:144], label="Targets", color="green")
    plt.plot(outputs[:144], label="Outputs", color="orange")
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
