
from .utils import *

from . import host

from .managed import Managed
from . import cache

from . import milk

from .url import from_url, query, url_of

from ._sardine import sync

__all__ = ['from_url', 'url_of', 'query',
           'ndarray',
           'int8', 'int16', 'int32', 'int64',
           'uint8', 'uint16', 'uint32', 'uint64',
           'float16', 'float32', 'float64', 'float128',
           'complex64', 'complex128', 'complex256',
           'host', 'milk', 'managed', 'sync', 'cache',
           'Managed'
]

__version__ = '1.0.0'
