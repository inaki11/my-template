import importlib
import os
import pkgutil

def get_loss(config):
    loss_name = config.name.lower()
    module_path = f"losses.{loss_name}"

    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError:
        available = list_available_modules("losses")
        raise ValueError(f"[Loss] '{loss_name}' no encontrado. Opciones válidas: {available}")

    if not hasattr(module, "build_loss"):
        raise AttributeError(f"[Loss] El módulo '{module_path}' no tiene una función 'build_loss(config)'")

    return module.build_loss(config)

def list_available_modules(package_name):
    package = importlib.import_module(package_name)
    package_path = package.__path__
    return sorted(
        name for _, name, ispkg in pkgutil.iter_modules(package_path) if not name.startswith("__")
    )
