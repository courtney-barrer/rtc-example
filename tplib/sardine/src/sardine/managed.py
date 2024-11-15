from typing import Any, Type, TypeVar, Union

from ._sardine import managed as mngd

T = TypeVar('T')

class Managed:
    """
    @brief Base class for managed
    """


    def __init__(self, managed_handle):
        self._managed_handle = managed_handle

    @property
    def named(self):
        return self._managed_handle.named

    def __repr__(self):
        return self._managed_handle.__repr__()

    @property
    def memory_available(self):
        return self._managed_handle.memory_available

    @property
    def memory_total(self):
        return self._managed_handle.memory_total

    @property
    def memory_used(self):
        return self._managed_handle.memory_used

    def open(self, type : Type[T], name: str) -> T:
        """
        @brief Opens a file and returns an object of the requested type.

        @param type The type to instantiate.
        @param args The arguments to pass to the constructor of the requested type.
        @param kwargs The keyword arguments to pass to the constructor of the requested type.
        @return An instance of the requested type.
        @throws TypeError if the requested type is not supported by the module.
        """
        return type.__shm_open__(self._managed_handle, name)

    def exists(self, type : Type[T], name: str) -> bool:
        """
        @brief Checks if a file exists.

        @param type The type to check.
        @param name The name of the file to check.
        @return True if the file exists, False otherwise.
        """
        return type.__shm_exists__(self._managed_handle, name)

    def destroy(self, type : Type[T], name: str) -> None:
        """
        @brief Destroys a file.

        @param type The type to destroy.
        @param name The name of the file to destroy.
        """
        type.__shm_destroy__(self._managed_handle, name)

    def create(self, type : Type[T], name: str, *args, **kwargs) -> T:
        """
        @brief Creates a file and returns an object of the requested type.

        @param type The type to instantiate.
        @param args The arguments to pass to the constructor of the requested type.
        @param kwargs The keyword arguments to pass to the constructor of the requested type.
        @return An instance of the requested type.
        @throws TypeError if the requested type is not supported by the module.
        """
        return type.__shm_create__(self._managed_handle, name, *args, **kwargs)

    def force_create(self, type : Type[T], name: str, *args, **kwargs) -> T:
        """
        @brief Creates a file and returns an object of the requested type.

        @param type The type to instantiate.
        @param args The arguments to pass to the constructor of the requested type.
        @param kwargs The keyword arguments to pass to the constructor of the requested type.
        @return An instance of the requested type.
        @throws TypeError if the requested type is not supported by the module.
        """
        return type.__shm_force_create__(self._managed_handle, name, *args, **kwargs)

    def open_or_create(self, type : Type[T], name: str, *args, **kwargs) -> T:
        """
        @brief Opens a file and returns an object of the requested type, creating it if it does not exist.

        @param type The type to instantiate.
        @param args The arguments to pass to the constructor of the requested type.
        @param kwargs The keyword arguments to pass to the constructor of the requested type.
        @return An instance of the requested type.
        @throws TypeError if the requested type is not supported by the module.
        """
        return type.__shm_open_or_create__(self._managed_handle, name, *args, **kwargs)


def open(name: str) -> Managed:
    """
    @brief Opens a managed object.

    @param name The name of the managed object to open.
    @return The opened managed object.
    """
    return Managed(mngd.open(name))

def create(name: str, file_size: int) -> Managed:
    """
    @brief Creates a managed object.

    @param name The name of the managed object to create.
    @param file_size The size of the managed object file.
    """
    return Managed(mngd.create(name, file_size))

def open_or_create(name: str, file_size: int) -> Managed:
    """
    @brief Opens or creates a managed object.

    @param name The name of the managed object to open or create.
    @param file_size The size of the managed object file.
    """
    return Managed(mngd.open_or_create(name, file_size))
