from typing import Any, Type, TypeVar, Union
from urllib.parse import ParseResult

from sardine import _sardine

from .utils import URLType

T = TypeVar('T')

# Dictionary of supported types as (module, type_name): (from_url_function, url_of_function)
# Lambdas are used to access `_sardine` functions only when needed, to avoid accessing non compiled functions
_supported_types = {
    ('numpy', 'ndarray'): (_sardine.numpy_ndarray_from_url, _sardine.url_of_numpy_ndarray),
}

# try import _sardinecuda and add it to supported types
try:
    from sardine import _sardinecuda
    _supported_types[('cupy', 'ndarray')] = (_sardinecuda.cupy_ndarray_from_url, _sardinecuda.url_of_cupy_ndarray)
except ImportError:
    pass


def _check_supported_type(type):
    """
    @brief Checks if a type is supported by the module and raises an exception if not.

    @param type The type to check.
    @throws TypeError if the type is not supported.
    """
    # Get module and type name of the input type
    module_name = type.__module__
    type_name = type.__name__
    # Raise TypeError if the type is not supported
    if (module_name, type_name) not in _supported_types:
        raise TypeError(f"The type '{module_name}.{type_name}' is not supported.")

def _get_type_functions(type):
    """
    @brief Retrieves the URL handling functions for a given type.

    @param type The type for which to retrieve URL functions.
    @return A tuple containing (from_url_function, url_of_function).
    @throws KeyError if the type is not supported.
    """
    # Extract module and type name of the requested type
    module_name = type.__module__
    type_name = type.__name__
    # Retrieve the corresponding functions from the dictionary
    return _supported_types[(module_name, type_name)]

def from_url(requested_type : Type[T], url : URLType) -> T:
    """
    @brief Instantiates an object of a given type from a URL.

    @param requested_type The type to instantiate.
    @param url The URL to use for instantiation.
    @return An instance of the requested type.
    @throws TypeError if the requested type is not supported by the module.
    """
    # Check if the type has a custom __from_url__ method
    if hasattr(requested_type, '__from_url__'):
        return requested_type.__from_url__(url)

    # Verify if the type is supported, raise otherwise
    _check_supported_type(requested_type)

    # Get the from_url function and use it to instantiate the object from the URL
    from_url_fn = _get_type_functions(requested_type)[0]

    return from_url_fn(url)


def url_of(value : Any, allow_local: bool = False) -> ParseResult:
    """
    @brief Generates a URL for a given value.

    @param value The value for which to generate a URL.
    @return The generated URL for the value.
    @throws TypeError if the type of the value is not supported by the module.
    """
    # Check if the value has a custom __url_of__ method
    if hasattr(value, '__url_of__'):
        return value.__url_of__()

    # Determine the type of the value
    requested_type = type(value)

    # Verify if the type is supported, raise otherwise
    _check_supported_type(requested_type)

    # Get the url_of function and use it to generate a URL for the value
    url_of_fn = _get_type_functions(requested_type)[1]
    return url_of_fn(value)
