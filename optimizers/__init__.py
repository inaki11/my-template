import importlib
import pkgutil

def get_optimizer(config, params):
    optim_name = config.name.lower()
    module_path = f"optimizers.{optim_name}"

    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError:
        available = list_available_modules("optimizers")
        raise ValueError(f"[Optimizer] '{optim_name}' no encontrado. Opciones válidas: {available}")

    if not hasattr(module, "build_optimizer"):
        raise AttributeError(f"[Optimizer] El módulo '{module_path}' no tiene una función 'build_optimizer(config, params)'")

    return module.build_optimizer(config, params)

def list_available_modules(package_name):
    package = importlib.import_module(package_name)
    package_path = package.__path__
    return sorted(
        name for _, name, ispkg in pkgutil.iter_modules(package_path) if not name.startswith("__")
    )
