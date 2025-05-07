import logging
import wandb


def wandb_img_output_errors(inputs, outputs, targets, classes_names):
    """
    Log evaluation errors to wandb.
    """
    examples = []
    preds = outputs.argmax(axis=1)
    for i in range(len(inputs)):
        img = wandb.Image(
            inputs[i],
            # apply softmax to outputs
            caption=f"Pred: {classes_names[preds[i]]},  True: {classes_names[targets[i]]}",
        )
        examples.append(img)

    wandb.log({"examples": examples})


def build_logger(config):
    return wandb_img_output_errors
