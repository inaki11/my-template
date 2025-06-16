import importlib
import os
import pkgutil

def get_model(input_size, output_size, config):
    model_name = config.name.lower()
    module_path = f"models.{model_name}"

    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError:
        available = list_available_modules("models")
        raise ValueError(f"[Model] '{model_name}' no encontrado. Opciones válidas: {available}")

    if not hasattr(module, "build_model"):
        raise AttributeError(f"[Model] El módulo '{module_path}' no tiene una función 'build_model(config)'")

    return module.build_model(input_size, output_size, config)

def list_available_modules(package_name):
    package = importlib.import_module(package_name)
    package_path = package.__path__
    return sorted(
        name for _, name, ispkg in pkgutil.iter_modules(package_path) if not ispkg and not name.startswith("__")
    )

