from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

from . import _version
__version__ = _version.get_versions()['version']

from .datasets import load_test, load_unformatted_example
from .neighbors import neighbor_analysis
