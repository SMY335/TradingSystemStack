"""
Unified logging configuration for TradingSystemStack.

Provides centralized logging setup with file and console handlers.
"""
import logging
import logging.handlers
from pathlib import Path
from typing import Optional
import sys


class ColoredFormatter(logging.Formatter):
    """Colored formatter for console output."""

    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m'       # Reset
    }

    def format(self, record):
        """Format record with colors."""
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = (
                f"{self.COLORS[levelname]}{levelname:8s}{self.COLORS['RESET']}"
            )
        return super().format(record)


def setup_logging(
    level: str = 'INFO',
    log_file: Optional[str] = None,
    log_dir: Optional[str] = 'logs',
    console: bool = True,
    colored: bool = True,
    format_str: Optional[str] = None,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> logging.Logger:
    """Setup unified logging configuration.

    Args:
        level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
        log_file: Log file name (None = no file logging)
        log_dir: Directory for log files
        console: Enable console output
        colored: Use colored output for console
        format_str: Custom format string
        max_bytes: Max bytes per log file (for rotation)
        backup_count: Number of backup files to keep

    Returns:
        Root logger instance

    Examples:
        >>> logger = setup_logging(level='DEBUG', log_file='trading.log')
        >>> logger.info("Application started")
    """
    # Get root logger
    logger = logging.getLogger()

    # Clear existing handlers
    logger.handlers.clear()

    # Set level
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(numeric_level)

    # Default format
    if format_str is None:
        format_str = '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'

    date_format = '%Y-%m-%d %H:%M:%S'

    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)

        if colored:
            console_formatter = ColoredFormatter(format_str, datefmt=date_format)
        else:
            console_formatter = logging.Formatter(format_str, datefmt=date_format)

        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    # File handler (with rotation)
    if log_file:
        if log_dir:
            log_path = Path(log_dir)
            log_path.mkdir(parents=True, exist_ok=True)
            log_file_path = log_path / log_file
        else:
            log_file_path = Path(log_file)

        file_handler = logging.handlers.RotatingFileHandler(
            log_file_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(numeric_level)

        file_formatter = logging.Formatter(format_str, datefmt=date_format)
        file_handler.setFormatter(file_formatter)

        logger.addHandler(file_handler)

        logger.info(f"Logging to file: {log_file_path}")

    logger.info(f"Logging initialized at level: {level}")

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get logger for module.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Logger instance

    Examples:
        >>> logger = get_logger(__name__)
        >>> logger.info("Module loaded")
    """
    return logging.getLogger(name)


def set_level(level: str, logger_name: Optional[str] = None) -> None:
    """Set logging level.

    Args:
        level: Logging level
        logger_name: Specific logger name (None = root logger)

    Examples:
        >>> set_level('DEBUG')  # All loggers
        >>> set_level('WARNING', 'src.indicators')  # Specific module
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    if logger_name:
        logger = logging.getLogger(logger_name)
    else:
        logger = logging.getLogger()

    logger.setLevel(numeric_level)

    # Also set handlers
    for handler in logger.handlers:
        handler.setLevel(numeric_level)


def disable_module_logging(module_name: str) -> None:
    """Disable logging for specific module.

    Args:
        module_name: Module to disable

    Examples:
        >>> disable_module_logging('urllib3')
        >>> disable_module_logging('matplotlib')
    """
    logging.getLogger(module_name).setLevel(logging.CRITICAL)


def enable_debug_for_module(module_name: str) -> None:
    """Enable debug logging for specific module.

    Args:
        module_name: Module to enable debug for

    Examples:
        >>> enable_debug_for_module('src.indicators')
    """
    logging.getLogger(module_name).setLevel(logging.DEBUG)


class LogContext:
    """Context manager for temporary logging level change.

    Examples:
        >>> with LogContext('DEBUG'):
        ...     # Debug logging active here
        ...     logger.debug("This will show")
        >>> # Back to previous level
    """

    def __init__(self, level: str, logger_name: Optional[str] = None):
        """Initialize context.

        Args:
            level: Temporary logging level
            logger_name: Specific logger (None = root)
        """
        self.level = level
        self.logger_name = logger_name
        self.original_level = None

    def __enter__(self):
        """Enter context - save current level and set new."""
        if self.logger_name:
            logger = logging.getLogger(self.logger_name)
        else:
            logger = logging.getLogger()

        self.original_level = logger.level
        numeric_level = getattr(logging, self.level.upper(), logging.INFO)
        logger.setLevel(numeric_level)

        return logger

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context - restore original level."""
        if self.logger_name:
            logger = logging.getLogger(self.logger_name)
        else:
            logger = logging.getLogger()

        logger.setLevel(self.original_level)


def log_function_call(func):
    """Decorator to log function calls.

    Examples:
        >>> @log_function_call
        >>> def my_function(x, y):
        ...     return x + y
    """
    import functools

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        logger.debug(f"Calling {func.__name__}({args}, {kwargs})")

        try:
            result = func(*args, **kwargs)
            logger.debug(f"{func.__name__} returned: {result}")
            return result
        except Exception as e:
            logger.error(f"{func.__name__} raised {type(e).__name__}: {e}")
            raise

    return wrapper


# Silence noisy libraries by default
def silence_noisy_loggers():
    """Silence commonly noisy loggers."""
    noisy = [
        'urllib3',
        'requests',
        'matplotlib',
        'PIL',
        'asyncio',
        'websockets',
        'httpx',
        'httpcore'
    ]

    for module in noisy:
        disable_module_logging(module)
