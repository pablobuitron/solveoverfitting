from typing import Type

from . import splits_generator

_REGISTRY: dict[str, Type[splits_generator.SplitsGenerator]] = {}


def register_splits_generator(sg: Type[splits_generator.SplitsGenerator]) -> Type[splits_generator.SplitsGenerator]:
    _REGISTRY[sg.__name__] = sg
    return sg


def from_dict(input_dict: dict):
    class_name = input_dict.get('class_name')
    if not class_name:
        raise ValueError('No class name provided inside the input dictionary')
    if class_name not in _REGISTRY:
        raise ValueError(f'Class name already not registered, '
                         f'please ensure you are using the decorator register_splits_generator inside {class_name}')

    class_ = _REGISTRY[class_name]
    return class_.from_dict(input_dict)
