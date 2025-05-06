import importlib
import pkgutil

def get_metrics(config_list):
    metrics = []
    for item in config_list:
        name = item['name'].lower()
        module_path = f"metrics.{name}"

        try:
            module = importlib.import_module(module_path)
        except ModuleNotFoundError:
            available = list_available()
            raise ValueError(f"[Metric] '{name}' no encontrada. Opciones válidas: {available}")

        if not hasattr(module, "build_metric"):
            raise AttributeError(f"[Metric] El módulo '{module_path}' no tiene una función 'build_metric(config)'")

        metrics.append(module.build_metric(item))
    return metrics

def list_available():
    import metrics
    return sorted(name for _, name, _ in pkgutil.iter_modules(metrics.__path__))
