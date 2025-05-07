import importlib
import pkgutil
import logging


def get_output_logger(config):
    logger_name = config.name.lower()
    module_path = f"loggers.{logger_name}"

    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError:
        available = list_available_modules("loggers")
        raise ValueError(
            f"[Logger] '{logger_name}' no encontrado. Opciones válidas: {available}"
        )

    if not hasattr(module, "build_logger"):
        raise AttributeError(
            f"[Logger] El módulo '{module_path}' no tiene una función 'build_logger(config)'"
        )

    return module.build_logger(config)


def list_available_modules(package_name):
    package = importlib.import_module(package_name)
    package_path = package.__path__
    return sorted(
        name
        for _, name, ispkg in pkgutil.iter_modules(package_path)
        if not name.startswith("__")
    )


def setup_logger():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    return logging.getLogger("trainer")
