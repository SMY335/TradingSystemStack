"""
Dynamic registry system for TradingSystemStack.

Provides registration and discovery of indicators, strategies, patterns, and scan operators.
"""
from typing import Dict, Any, Callable, Optional, List, Type
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


class RegistryError(Exception):
    """Custom exception for registry operations."""
    pass


@dataclass
class RegistryEntry:
    """Entry in the registry.

    Attributes:
        name: Canonical name
        obj: Registered object (class, function, etc.)
        aliases: Alternative names
        category: Entry category (e.g., 'indicator', 'strategy')
        metadata: Additional metadata
        params_schema: Parameter schema for validation
    """
    name: str
    obj: Any
    aliases: List[str] = field(default_factory=list)
    category: str = ''
    metadata: Dict[str, Any] = field(default_factory=dict)
    params_schema: Optional[Dict[str, Any]] = None


class Registry:
    """Dynamic registry for components.

    Manages registration and lookup of indicators, strategies, patterns, etc.

    Examples:
        >>> registry = Registry()
        >>> registry.register('RSI', RSIIndicator, category='indicator')
        >>> rsi = registry.get('RSI')
        >>> all_indicators = registry.list_by_category('indicator')
    """

    def __init__(self):
        """Initialize registry."""
        self._entries: Dict[str, RegistryEntry] = {}
        self._alias_map: Dict[str, str] = {}  # alias → canonical name
        logger.debug("Registry initialized")

    def register(
        self,
        name: str,
        obj: Any,
        aliases: Optional[List[str]] = None,
        category: str = '',
        metadata: Optional[Dict[str, Any]] = None,
        params_schema: Optional[Dict[str, Any]] = None,
        overwrite: bool = False
    ) -> None:
        """Register an object in the registry.

        Args:
            name: Canonical name
            obj: Object to register
            aliases: Alternative names
            category: Entry category
            metadata: Additional metadata
            params_schema: Parameter schema
            overwrite: Allow overwriting existing entry

        Raises:
            RegistryError: If name already registered and overwrite=False

        Examples:
            >>> registry.register(
            ...     'RSI',
            ...     RSIIndicator,
            ...     aliases=['rsi', 'RelativeStrengthIndex'],
            ...     category='indicator',
            ...     params_schema={'period': {'type': 'int', 'min': 2, 'max': 100}}
            ... )
        """
        if name in self._entries and not overwrite:
            raise RegistryError(f"Name already registered: {name}")

        aliases = aliases or []
        metadata = metadata or {}

        # Check alias conflicts
        for alias in aliases:
            if alias in self._alias_map and not overwrite:
                existing = self._alias_map[alias]
                raise RegistryError(
                    f"Alias '{alias}' already mapped to '{existing}'"
                )

        # Create entry
        entry = RegistryEntry(
            name=name,
            obj=obj,
            aliases=aliases,
            category=category,
            metadata=metadata,
            params_schema=params_schema
        )

        # Register
        self._entries[name] = entry

        # Register aliases
        for alias in aliases:
            self._alias_map[alias] = name

        logger.debug(
            f"Registered '{name}' in category '{category}' "
            f"with {len(aliases)} aliases"
        )

    def get(self, name: str, default: Any = None) -> Any:
        """Get registered object by name or alias.

        Args:
            name: Name or alias to lookup
            default: Default value if not found

        Returns:
            Registered object or default

        Examples:
            >>> rsi_indicator = registry.get('RSI')
            >>> rsi_indicator = registry.get('rsi')  # Using alias
        """
        # Try canonical name first
        if name in self._entries:
            return self._entries[name].obj

        # Try alias
        if name in self._alias_map:
            canonical = self._alias_map[name]
            return self._entries[canonical].obj

        # Not found
        if default is not None:
            return default

        raise RegistryError(f"Not found in registry: {name}")

    def get_entry(self, name: str) -> RegistryEntry:
        """Get full registry entry (includes metadata).

        Args:
            name: Name or alias to lookup

        Returns:
            RegistryEntry object

        Examples:
            >>> entry = registry.get_entry('RSI')
            >>> print(entry.params_schema)
        """
        # Try canonical name first
        if name in self._entries:
            return self._entries[name]

        # Try alias
        if name in self._alias_map:
            canonical = self._alias_map[name]
            return self._entries[canonical]

        raise RegistryError(f"Not found in registry: {name}")

    def exists(self, name: str) -> bool:
        """Check if name exists in registry.

        Args:
            name: Name or alias to check

        Returns:
            True if exists

        Examples:
            >>> registry.exists('RSI')
            True
            >>> registry.exists('rsi')  # Alias
            True
        """
        return name in self._entries or name in self._alias_map

    def unregister(self, name: str) -> None:
        """Remove entry from registry.

        Args:
            name: Canonical name to remove

        Raises:
            RegistryError: If name not found

        Examples:
            >>> registry.unregister('RSI')
        """
        if name not in self._entries:
            raise RegistryError(f"Not found in registry: {name}")

        entry = self._entries[name]

        # Remove aliases
        for alias in entry.aliases:
            if alias in self._alias_map:
                del self._alias_map[alias]

        # Remove entry
        del self._entries[name]

        logger.debug(f"Unregistered '{name}'")

    def list_all(self) -> List[str]:
        """List all registered names.

        Returns:
            List of canonical names

        Examples:
            >>> names = registry.list_all()
            >>> print(names)
            ['RSI', 'MACD', 'EMA', ...]
        """
        return sorted(self._entries.keys())

    def list_by_category(self, category: str) -> List[str]:
        """List entries in a category.

        Args:
            category: Category to filter by

        Returns:
            List of names in category

        Examples:
            >>> indicators = registry.list_by_category('indicator')
            >>> strategies = registry.list_by_category('strategy')
        """
        return sorted([
            name for name, entry in self._entries.items()
            if entry.category == category
        ])

    def get_categories(self) -> List[str]:
        """Get all unique categories.

        Returns:
            List of category names

        Examples:
            >>> categories = registry.get_categories()
            >>> print(categories)
            ['indicator', 'strategy', 'pattern', 'operator']
        """
        categories = set(entry.category for entry in self._entries.values())
        return sorted(categories - {''})  # Exclude empty category

    def get_aliases(self, name: str) -> List[str]:
        """Get all aliases for a name.

        Args:
            name: Name to lookup

        Returns:
            List of aliases

        Examples:
            >>> aliases = registry.get_aliases('RSI')
            >>> print(aliases)
            ['rsi', 'RelativeStrengthIndex']
        """
        entry = self.get_entry(name)
        return entry.aliases

    def get_params_schema(self, name: str) -> Optional[Dict[str, Any]]:
        """Get parameter schema for an entry.

        Args:
            name: Name to lookup

        Returns:
            Parameter schema or None

        Examples:
            >>> schema = registry.get_params_schema('RSI')
            >>> print(schema)
            {'period': {'type': 'int', 'min': 2, 'max': 100, 'default': 14}}
        """
        entry = self.get_entry(name)
        return entry.params_schema

    def search(self, query: str) -> List[str]:
        """Search registry by name/alias substring.

        Args:
            query: Search query (case-insensitive)

        Returns:
            List of matching names

        Examples:
            >>> results = registry.search('MA')
            >>> print(results)
            ['SMA', 'EMA', 'MACD', 'VWMA']
        """
        query_lower = query.lower()
        matches = set()

        # Search canonical names
        for name in self._entries.keys():
            if query_lower in name.lower():
                matches.add(name)

        # Search aliases
        for alias, canonical in self._alias_map.items():
            if query_lower in alias.lower():
                matches.add(canonical)

        return sorted(matches)

    def clear(self, category: Optional[str] = None) -> None:
        """Clear registry entries.

        Args:
            category: If provided, only clear this category. Otherwise clear all.

        Examples:
            >>> registry.clear('indicator')  # Clear only indicators
            >>> registry.clear()  # Clear everything
        """
        if category is None:
            # Clear all
            self._entries.clear()
            self._alias_map.clear()
            logger.debug("Cleared all registry entries")
        else:
            # Clear specific category
            to_remove = [
                name for name, entry in self._entries.items()
                if entry.category == category
            ]
            for name in to_remove:
                self.unregister(name)
            logger.debug(f"Cleared {len(to_remove)} entries from category '{category}'")

    def __len__(self) -> int:
        """Get number of registered entries."""
        return len(self._entries)

    def __contains__(self, name: str) -> bool:
        """Check if name exists in registry (supports 'in' operator)."""
        return self.exists(name)

    def __repr__(self) -> str:
        """String representation."""
        return f"Registry(entries={len(self._entries)}, categories={len(self.get_categories())})"


# Global registries for different component types
_indicator_registry = Registry()
_strategy_registry = Registry()
_pattern_registry = Registry()
_operator_registry = Registry()
_loader_registry = Registry()


def get_indicator_registry() -> Registry:
    """Get global indicator registry."""
    return _indicator_registry


def get_strategy_registry() -> Registry:
    """Get global strategy registry."""
    return _strategy_registry


def get_pattern_registry() -> Registry:
    """Get global pattern registry."""
    return _pattern_registry


def get_operator_registry() -> Registry:
    """Get global operator registry (for scanner DSL)."""
    return _operator_registry


def get_loader_registry() -> Registry:
    """Get global data loader registry."""
    return _loader_registry


# Decorator for easy registration
def register_indicator(
    name: str,
    aliases: Optional[List[str]] = None,
    params_schema: Optional[Dict[str, Any]] = None
):
    """Decorator to register indicator.

    Args:
        name: Indicator name
        aliases: Alternative names
        params_schema: Parameter schema

    Examples:
        >>> @register_indicator('RSI', aliases=['rsi'], params_schema={'period': {...}})
        >>> class RSIIndicator:
        ...     pass
    """
    def decorator(cls):
        _indicator_registry.register(
            name,
            cls,
            aliases=aliases,
            category='indicator',
            params_schema=params_schema
        )
        return cls
    return decorator


def register_strategy(
    name: str,
    aliases: Optional[List[str]] = None,
    params_schema: Optional[Dict[str, Any]] = None
):
    """Decorator to register strategy.

    Examples:
        >>> @register_strategy('EMA_Cross', aliases=['ema_crossover'])
        >>> class EMACrossStrategy:
        ...     pass
    """
    def decorator(cls):
        _strategy_registry.register(
            name,
            cls,
            aliases=aliases,
            category='strategy',
            params_schema=params_schema
        )
        return cls
    return decorator
