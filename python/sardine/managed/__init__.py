from .._sardine import managed

import builtins
import numpy as np

# class numpy_proxy:
#     def __init__(self, impl):
#         self.__impl = impl

#     def __getitem__(self, nb_element):
#         return

    # def open(self, name, dtype):
    #     return self.proxy_bytes.open(name).view(dtype = dtype)

    # def create(self, name, dtype, size):
    #     return self.proxy_bytes.create(name, size * np.dtype(dtype).itemsize).view(dtype = dtype)

    # def open_or_create(self, name, dtype, size):
    #     return self.proxy_bytes.open_or_create(name, size * np.dtype(dtype).itemsize).view(dtype = dtype)

    # def force_create(self, name, *args, **kwargs):
    #     return self.proxy_bytes.force_create(name, size * np.dtype(dtype).itemsize).view(dtype = dtype)

    # def set(self, name, data):
    #     data = np.array(data)
    #     self.open(name, data.dtype)[:] = data

    # def exist(self, name):
    #     return self.proxy_bytes.exist(name)

    # def destroy(self, name):
    #     return self.proxy_bytes.destroy(name)

class Managed:
    def __init__(self, impl):
        self.__impl = impl

    def _proxy(self, vtype):
        match(vtype):
            # case managed.type_aware: return vtype.__managed__proxy

            case builtins.bool:  return self.__impl.proxy_bool

            case builtins.int:   return self.__impl.proxy_int64
            case builtins.float: return self.__impl.proxy_double

            case np.float16:     return self.__impl.proxy_half
            case np.float32:     return self.__impl.proxy_float
            case np.float64:     return self.__impl.proxy_double

            case np.int8:         return self.__impl.proxy_int8
            case np.int16:        return self.__impl.proxy_int16
            case np.int32:        return self.__impl.proxy_int32
            case np.int64:        return self.__impl.proxy_int64

            case np.uint8:        return self.__impl.proxy_uint8
            case np.uint16:       return self.__impl.proxy_uint16
            case np.uint32:       return self.__impl.proxy_uint32
            case np.uint64:       return self.__impl.proxy_uint64

            case np.array:        return numpy_proxy(self.__impl)

            case _: raise 'unknown type !'


    def open(self, vtype, name):
        return self._proxy(vtype).open(name)

    def create(self, vtype, name, *args, **kwargs):
        return self._proxy(vtype).create(name, *args, **kwargs)

    def open_or_create(self, vtype, name, *args, **kwargs):
        return self._proxy(vtype).open_or_create(name, *args, **kwargs)

    def force_create(self, vtype, name, *args, **kwargs):
        return self._proxy(vtype).force_create(name, *args, **kwargs)

    def set(self, vtype, name, *args, **kwargs):
        return self._proxy(vtype).set(name, *args, **kwargs)

    def exist(self, vtype, name):
        return self._proxy(vtype).exist(name)

    def destroy(self, vtype, name):
        return self._proxy(vtype).destroy(name)

def open_or_create(name, size):
    return Managed(managed.open_or_create(name, size))
