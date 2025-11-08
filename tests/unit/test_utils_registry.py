"""
Unit tests for utils.registry module.
"""
import pytest

from src.utils.registry import (
    Registry,
    RegistryEntry,
    RegistryError,
    register_indicator,
    register_strategy,
)


class DummyIndicator:
    """Dummy indicator for testing."""
    def __init__(self, period=14):
        self.period = period


class DummyStrategy:
    """Dummy strategy for testing."""
    def __init__(self, fast=12, slow=26):
        self.fast = fast
        self.slow = slow


class TestRegistry:
    """Test Registry class."""

    def setup_method(self):
        """Create fresh registry for each test."""
        self.registry = Registry()

    def test_register_and_get(self):
        """Test basic registration and retrieval."""
        self.registry.register('RSI', DummyIndicator, category='indicator')

        retrieved = self.registry.get('RSI')
        assert retrieved == DummyIndicator

    def test_register_with_aliases(self):
        """Test registration with aliases."""
        self.registry.register(
            'RSI',
            DummyIndicator,
            aliases=['rsi', 'RelativeStrengthIndex'],
            category='indicator'
        )

        # Should retrieve by canonical name
        assert self.registry.get('RSI') == DummyIndicator

        # Should retrieve by aliases
        assert self.registry.get('rsi') == DummyIndicator
        assert self.registry.get('RelativeStrengthIndex') == DummyIndicator

    def test_register_duplicate_error(self):
        """Test error when registering duplicate name."""
        self.registry.register('RSI', DummyIndicator)

        with pytest.raises(RegistryError, match="already registered"):
            self.registry.register('RSI', DummyStrategy)

    def test_register_duplicate_overwrite(self):
        """Test overwriting existing registration."""
        self.registry.register('RSI', DummyIndicator)
        self.registry.register('RSI', DummyStrategy, overwrite=True)

        retrieved = self.registry.get('RSI')
        assert retrieved == DummyStrategy

    def test_get_not_found(self):
        """Test getting non-existent entry."""
        with pytest.raises(RegistryError, match="Not found"):
            self.registry.get('NonExistent')

    def test_get_with_default(self):
        """Test getting with default value."""
        result = self.registry.get('NonExistent', default='default_value')
        assert result == 'default_value'

    def test_get_entry(self):
        """Test getting full registry entry."""
        metadata = {'source': 'talib', 'version': '1.0'}
        params_schema = {'period': {'type': 'int', 'min': 1, 'max': 100}}

        self.registry.register(
            'RSI',
            DummyIndicator,
            aliases=['rsi'],
            category='indicator',
            metadata=metadata,
            params_schema=params_schema
        )

        entry = self.registry.get_entry('RSI')

        assert isinstance(entry, RegistryEntry)
        assert entry.name == 'RSI'
        assert entry.obj == DummyIndicator
        assert entry.aliases == ['rsi']
        assert entry.category == 'indicator'
        assert entry.metadata == metadata
        assert entry.params_schema == params_schema

    def test_exists(self):
        """Test checking if entry exists."""
        self.registry.register('RSI', DummyIndicator, aliases=['rsi'])

        assert self.registry.exists('RSI') is True
        assert self.registry.exists('rsi') is True  # Alias
        assert self.registry.exists('NonExistent') is False

    def test_exists_with_in_operator(self):
        """Test 'in' operator."""
        self.registry.register('RSI', DummyIndicator)

        assert 'RSI' in self.registry
        assert 'NonExistent' not in self.registry

    def test_unregister(self):
        """Test unregistering entry."""
        self.registry.register('RSI', DummyIndicator, aliases=['rsi'])

        assert self.registry.exists('RSI')
        assert self.registry.exists('rsi')

        self.registry.unregister('RSI')

        assert not self.registry.exists('RSI')
        assert not self.registry.exists('rsi')

    def test_unregister_not_found(self):
        """Test error when unregistering non-existent entry."""
        with pytest.raises(RegistryError, match="Not found"):
            self.registry.unregister('NonExistent')

    def test_list_all(self):
        """Test listing all entries."""
        self.registry.register('RSI', DummyIndicator)
        self.registry.register('MACD', DummyStrategy)
        self.registry.register('EMA', DummyIndicator)

        all_names = self.registry.list_all()

        assert all_names == ['EMA', 'MACD', 'RSI']  # Sorted

    def test_list_by_category(self):
        """Test listing entries by category."""
        self.registry.register('RSI', DummyIndicator, category='indicator')
        self.registry.register('MACD', DummyIndicator, category='indicator')
        self.registry.register('EMAStrategy', DummyStrategy, category='strategy')

        indicators = self.registry.list_by_category('indicator')
        strategies = self.registry.list_by_category('strategy')

        assert indicators == ['MACD', 'RSI']
        assert strategies == ['EMAStrategy']

    def test_get_categories(self):
        """Test getting all categories."""
        self.registry.register('RSI', DummyIndicator, category='indicator')
        self.registry.register('EMAStrategy', DummyStrategy, category='strategy')
        self.registry.register('Triangle', DummyIndicator, category='pattern')

        categories = self.registry.get_categories()

        assert categories == ['indicator', 'pattern', 'strategy']  # Sorted

    def test_get_aliases(self):
        """Test getting aliases for entry."""
        self.registry.register('RSI', DummyIndicator, aliases=['rsi', 'RelativeStrength'])

        aliases = self.registry.get_aliases('RSI')
        assert aliases == ['rsi', 'RelativeStrength']

    def test_get_params_schema(self):
        """Test getting parameter schema."""
        schema = {'period': {'type': 'int', 'min': 1, 'max': 100}}
        self.registry.register('RSI', DummyIndicator, params_schema=schema)

        retrieved_schema = self.registry.get_params_schema('RSI')
        assert retrieved_schema == schema

    def test_search(self):
        """Test searching registry."""
        self.registry.register('RSI', DummyIndicator, aliases=['rsi'])
        self.registry.register('MACD', DummyStrategy, aliases=['macd'])
        self.registry.register('EMA', DummyIndicator, aliases=['ema'])
        self.registry.register('SMA', DummyIndicator)

        # Search for 'MA'
        results = self.registry.search('MA')
        assert 'MACD' in results
        assert 'EMA' in results
        assert 'SMA' in results
        assert 'RSI' not in results

    def test_clear_all(self):
        """Test clearing all entries."""
        self.registry.register('RSI', DummyIndicator)
        self.registry.register('MACD', DummyStrategy)

        assert len(self.registry) == 2

        self.registry.clear()

        assert len(self.registry) == 0

    def test_clear_by_category(self):
        """Test clearing specific category."""
        self.registry.register('RSI', DummyIndicator, category='indicator')
        self.registry.register('MACD', DummyIndicator, category='indicator')
        self.registry.register('EMAStrategy', DummyStrategy, category='strategy')

        assert len(self.registry) == 3

        self.registry.clear(category='indicator')

        assert len(self.registry) == 1
        assert self.registry.exists('EMAStrategy')

    def test_len(self):
        """Test len() operator."""
        assert len(self.registry) == 0

        self.registry.register('RSI', DummyIndicator)
        assert len(self.registry) == 1

        self.registry.register('MACD', DummyStrategy)
        assert len(self.registry) == 2

    def test_repr(self):
        """Test string representation."""
        repr_str = repr(self.registry)
        assert 'Registry' in repr_str
        assert 'entries=' in repr_str


class TestRegistryDecorators:
    """Test registry decorators."""

    def test_register_indicator_decorator(self):
        """Test @register_indicator decorator."""
        @register_indicator('TestIndicator', aliases=['test_ind'])
        class TestIndicator:
            pass

        from src.utils.registry import get_indicator_registry
        registry = get_indicator_registry()

        assert registry.exists('TestIndicator')
        assert registry.exists('test_ind')

    def test_register_strategy_decorator(self):
        """Test @register_strategy decorator."""
        @register_strategy('TestStrategy', aliases=['test_strat'])
        class TestStrategy:
            pass

        from src.utils.registry import get_strategy_registry
        registry = get_strategy_registry()

        assert registry.exists('TestStrategy')
        assert registry.exists('test_strat')
