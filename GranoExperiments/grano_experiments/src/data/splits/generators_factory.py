from typing import Dict, Type, Any

# Registro global de generadores de splits
_REGISTRY: Dict[str, Type[Any]] = {}


def register_splits_generator(cls: Type[Any]) -> Type[Any]:
    """
    Decorador para registrar un generador de splits.

    El nombre de clase (cls.__name__) se usará como 'class_name' en los
    diccionarios serializados que guarda Experiment.
    """
    _REGISTRY[cls.__name__] = cls
    return cls


def from_dict(d: dict) -> Any:
    """
    Reconstruye un generador de splits a partir de un diccionario.

    Formato esperado:
      {
        "class_name": "PerFieldPointShuffleSplitGenerator",
        "init_args": {
            "train_ratio": 0.7,
            "validation_ratio": 0.15,
            "test_ratio": 0.15,
            "seed": 403
        }
      }
    """
    class_name = d.get("class_name")
    if class_name is None:
        raise ValueError("Missing 'class_name' in splits generator description")

    cls = _REGISTRY.get(class_name)
    if cls is None:
        raise ValueError(
            f"Class name already not registered, please ensure you are using "
            f"the decorator register_splits_generator inside {class_name}"
        )

    init_args = d.get("init_args", {}) or {}
    return cls(**init_args)
