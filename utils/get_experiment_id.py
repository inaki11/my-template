import hashlib
import json
from omegaconf import OmegaConf


def get_experiment_id(config):
    cfg_str = json.dumps(OmegaConf.to_container(config, resolve=True), sort_keys=True)
    short_hash = hashlib.md5(cfg_str.encode()).hexdigest()[:8]
    return f"{config.dataset.name}_{config.model.name}_{short_hash}"  # TO DO: Cambiar por un nombre más descriptivo
