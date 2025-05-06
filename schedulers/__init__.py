import importlib
import pkgutil

def get_scheduler(config, optimizer):
    sched_name = config.name.lower()
    module_path = f"schedulers.{sched_name}"

    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError:
        available = list_available_modules("schedulers")
        raise ValueError(f"[Scheduler] '{sched_name}' no encontrado. Opciones válidas: {available}")

    if not hasattr(module, "build_scheduler"):
        raise AttributeError(f"[Scheduler] El módulo '{module_path}' no tiene una función 'build_scheduler(config, optimizer)'")

    return module.build_scheduler(config, optimizer)

def list_available_modules(package_name):
    package = importlib.import_module(package_name)
    package_path = package.__path__
    return sorted(
        name for _, name, ispkg in pkgutil.iter_modules(package_path) if not name.startswith("__")
    )
