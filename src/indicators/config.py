"""
Global configuration for indicators module.
"""
from typing import Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class IndicatorConfig:
    """Global configuration for indicators.

    Attributes:
        fallback_enabled: Enable fallback to alternative libraries
        cache_enabled: Enable result caching
        validation_strict: Strict validation mode
        preferred_libraries: Order of preference for libraries
    """
    fallback_enabled: bool = True
    cache_enabled: bool = False
    validation_strict: bool = True
    preferred_libraries: List[str] = field(default_factory=lambda: [
        'talib',
        'pandas_ta',
        'ta',
        'smartmoneyconcepts',
        'vectorbt'
    ])


# Global config instance
_config = IndicatorConfig()


def get_config() -> IndicatorConfig:
    """Get global indicator configuration."""
    return _config


def set_config(**kwargs) -> None:
    """Update global configuration.

    Args:
        **kwargs: Configuration parameters to update
    """
    global _config
    for key, value in kwargs.items():
        if hasattr(_config, key):
            setattr(_config, key, value)
