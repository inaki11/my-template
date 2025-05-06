import importlib
import pkgutil
from .compose import Compose as Compose

def get_preprocess_transforms(preproc_config_list):
    transform_list = []
    for item in preproc_config_list:
        name = item['name'].lower()
        module_path = f"data.preprocessing.{name}"
        try:
            module = importlib.import_module(module_path)
        except ModuleNotFoundError:
            available = list_available()
            raise ValueError(f"[Preprocessing] '{name}' no encontrado. Opciones: {available}")

        if not hasattr(module, "build_transform"):
            raise AttributeError(f"Falta 'build_transform(config)' en {module_path}")
        
        transform_list.append(module.build_transform(item))

    return Compose(transform_list)

def list_available():
    import data.preprocessing
    return sorted(name for _, name, _ in pkgutil.iter_modules(data.preprocessing.__path__))
