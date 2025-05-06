import importlib
import pkgutil

def get_callbacks(config_list):
    callbacks = []
    for item in config_list:
        name = item['name'].lower()
        module_path = f"callbacks.{name}"

        try:
            module = importlib.import_module(module_path)
        except ModuleNotFoundError:
            available = list_available()
            raise ValueError(f"[Callback] '{name}' no encontrado. Opciones: {available}")

        if not hasattr(module, "build_callback"):
            raise AttributeError(f"Falta 'build_callback(config)' en {module_path}")

        callbacks.append(module.build_callback(item))
    return callbacks

def list_available():
    import callbacks
    return sorted(name for _, name, _ in pkgutil.iter_modules(callbacks.__path__))
