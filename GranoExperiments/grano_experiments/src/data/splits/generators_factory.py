from __future__ import annotations

from typing import Any, Dict, Type

# Registro global de generadores
_REGISTRY: Dict[str, Type[Any]] = {}


def register_splits_generator(cls: Type[Any]) -> Type[Any]:
    """
    Decorador para registrar un generador de splits.

    Se usa así:

        @register_splits_generator
        class RandomSplitGenerator(SplitsGenerator):
            ...

    Luego podemos reconstruirlo desde un diccionario con from_dict.
    """
    _REGISTRY[cls.__name__] = cls
    return cls


def from_dict(config: Dict[str, Any]):
    """
    Reconstruye un generador de splits a partir del diccionario guardado
    en el Experiment (self.experiment.splits_generator_data).

    Espera al menos:
        config["class_name"]: nombre de la clase registrada
    """
    class_name = config.get("class_name", None)
    if class_name is None:
        raise ValueError(f'config["class_name"] ausente en {config}')

    cls = _REGISTRY.get(class_name)
    if cls is None:
        raise ValueError(
            f'Class name "{class_name}" not registered. '
            f"Available: {list(_REGISTRY.keys())}"
        )

    # Si la clase define from_dict, la usamos; si no, construimos sin args.
    if hasattr(cls, "from_dict") and callable(getattr(cls, "from_dict")):
        return cls.from_dict(config)

    return cls()
