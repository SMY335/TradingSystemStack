"""
Custom exceptions for indicators module.
"""


class IndicatorError(Exception):
    """Base exception for indicator operations."""
    pass


class IndicatorNotFoundError(IndicatorError):
    """Raised when indicator is not found in registry."""
    pass


class InvalidParametersError(IndicatorError):
    """Raised when indicator parameters are invalid."""
    pass


class InvalidDataError(IndicatorError):
    """Raised when input data is invalid for indicator."""
    pass


class CalculationError(IndicatorError):
    """Raised when indicator calculation fails."""
    pass


class LibraryNotAvailableError(IndicatorError):
    """Raised when required library is not installed."""
    pass
