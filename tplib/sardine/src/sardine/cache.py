from typing import Any, Type, TypeVar, Union

T = TypeVar('T')

def create(type : Type[T], *args, **kwargs) -> T:
    """
    @brief Creates a file and returns an object of the requested type.

    @param type The type to instantiate.
    @param args The arguments to pass to the constructor of the requested type.
    @param kwargs The keyword arguments to pass to the constructor of the requested type.
    @return An instance of the requested type.
    @throws TypeError if the requested type is not supported by the module.
    """
    return type.__cache_create__(*args, **kwargs)
