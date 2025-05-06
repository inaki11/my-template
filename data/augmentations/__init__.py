# data/augmentations/__init__.py
import importlib
import pkgutil
from .compose import Compose as Compose

def get_augmentation_transforms(aug_config_list, split):
    transform_list = []
    for item in aug_config_list:
        name = item['name'].lower()
        module_path = f"data.augmentations.{name}"
        try:
            module = importlib.import_module(module_path)
        except ModuleNotFoundError:
            available = list_available()
            raise ValueError(f"[Augmentations] '{name}' no encontrado. Opciones: {available}")

        if not hasattr(module, "build_transform"):
            raise AttributeError(f"Falta 'build_transform(config, split)' en {module_path}")
        
        transform_list.append(module.build_transform(item, split=split))

    return Compose(transform_list)

def list_available():
    import data.augmentations
    return sorted(name for _, name, _ in pkgutil.iter_modules(data.augmentations.__path__))
