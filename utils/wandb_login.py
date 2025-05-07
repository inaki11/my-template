import json
import wandb


def wandb_login():
    with open("wandb/login.json") as f:
        wandb_key = json.load(f)["key"]
        wandb.login(key=wandb_key)
