import importlib
import pkgutil

def get_scaler(config):
    scaler_name = config.name.lower()
    module_path = f"data.scalers.{scaler_name}"

    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError:
        available = list_available_modules("data.scalers")
        raise ValueError(f"[Scaler] '{scaler_name}' no encontrado. Opciones válidas: {available}")

    if not hasattr(module, "build_scaler"):
        raise AttributeError(f"[Model] El módulo '{module_path}' no tiene una función 'build_scaler(config)'")

    return module.build_scaler(config)

def list_available_modules(package_name):
    package = importlib.import_module(package_name)
    package_path = package.__path__
    return sorted(
        name for _, name, ispkg in pkgutil.iter_modules(package_path) if not ispkg and not name.startswith("__")
    )

