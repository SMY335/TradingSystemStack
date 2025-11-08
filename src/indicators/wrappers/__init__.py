"""
Indicator wrappers for various libraries.
"""

# Import wrappers to trigger registration
try:
    from . import talib_wrapper
except ImportError:
    pass

try:
    from . import pandas_ta_wrapper
except ImportError:
    pass

__all__ = []
