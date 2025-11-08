"""
Unit tests for data.cache module.
"""
import pytest
import pandas as pd
import time
from pathlib import Path
from datetime import datetime, timedelta

from src.data.cache import DataCache, get_cache, CacheError


@pytest.fixture
def temp_cache_dir(tmp_path):
    """Create temporary cache directory."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    return cache_dir


@pytest.fixture
def cache(temp_cache_dir):
    """Create DataCache instance for testing."""
    return DataCache(cache_dir=temp_cache_dir, default_ttl_hours=24)


@pytest.fixture
def sample_df():
    """Create sample DataFrame for testing."""
    dates = pd.date_range('2024-01-01', periods=5, freq='D')
    return pd.DataFrame({
        'open': [100, 101, 102, 103, 104],
        'high': [102, 103, 104, 105, 106],
        'low': [99, 100, 101, 102, 103],
        'close': [101, 102, 103, 104, 105],
        'volume': [1000, 2000, 3000, 4000, 5000]
    }, index=dates)


class TestDataCacheBasics:
    """Test basic DataCache operations."""

    def test_cache_initialization(self, temp_cache_dir):
        """Test cache initialization."""
        cache = DataCache(cache_dir=temp_cache_dir)

        assert cache.cache_dir.exists()
        assert cache.default_ttl_hours == 24
        assert cache.compression == 'snappy'

    def test_set_and_get(self, cache, sample_df):
        """Test basic set and get operations."""
        cache.set('test_key', sample_df, ttl_hours=24)

        retrieved = cache.get('test_key')

        assert retrieved is not None
        assert len(retrieved) == len(sample_df)
        pd.testing.assert_frame_equal(retrieved, sample_df, check_freq=False)

    def test_get_nonexistent(self, cache):
        """Test getting non-existent key."""
        result = cache.get('nonexistent')

        assert result is None

    def test_get_with_default(self, cache):
        """Test getting with default value."""
        default_df = pd.DataFrame({'col': [1, 2, 3]})
        result = cache.get('nonexistent', default=default_df)

        pd.testing.assert_frame_equal(result, default_df)

    def test_exists(self, cache, sample_df):
        """Test exists method."""
        assert cache.exists('test_key') is False

        cache.set('test_key', sample_df)

        assert cache.exists('test_key') is True

    def test_contains_operator(self, cache, sample_df):
        """Test 'in' operator."""
        cache.set('test_key', sample_df)

        assert 'test_key' in cache
        assert 'nonexistent' not in cache

    def test_clear_specific_key(self, cache, sample_df):
        """Test clearing specific cache entry."""
        cache.set('key1', sample_df)
        cache.set('key2', sample_df)

        assert cache.exists('key1')
        assert cache.exists('key2')

        cache.clear('key1')

        assert not cache.exists('key1')
        assert cache.exists('key2')

    def test_clear_all(self, cache, sample_df):
        """Test clearing entire cache."""
        cache.set('key1', sample_df)
        cache.set('key2', sample_df)

        assert len(cache) == 2

        cache.clear()

        assert len(cache) == 0

    def test_list_keys(self, cache, sample_df):
        """Test listing all cache keys."""
        cache.set('key1', sample_df)
        cache.set('key2', sample_df)
        cache.set('key3', sample_df)

        keys = cache.list_keys()

        assert len(keys) == 3
        assert 'key1' in keys
        assert 'key2' in keys
        assert 'key3' in keys

    def test_len(self, cache, sample_df):
        """Test len() operator."""
        assert len(cache) == 0

        cache.set('key1', sample_df)
        assert len(cache) == 1

        cache.set('key2', sample_df)
        assert len(cache) == 2


class TestCacheTTL:
    """Test cache TTL (time-to-live) functionality."""

    def test_ttl_not_expired(self, cache, sample_df):
        """Test that non-expired cache returns data."""
        cache.set('test_key', sample_df, ttl_hours=24)

        # Immediately retrieve (should not be expired)
        result = cache.get('test_key')

        assert result is not None

    def test_ttl_expired(self, cache, sample_df):
        """Test that expired cache returns None."""
        # Set with very short TTL (simulate by manipulating metadata)
        cache.set('test_key', sample_df, ttl_hours=24)

        # Manually expire the cache
        expires_at = datetime.now() - timedelta(hours=1)
        cache.metadata['test_key']['expires_at'] = expires_at.isoformat()

        # Should return None (expired)
        result = cache.get('test_key')

        assert result is None

    def test_clear_expired(self, cache, sample_df):
        """Test clearing expired entries."""
        # Set multiple entries
        cache.set('key1', sample_df, ttl_hours=24)
        cache.set('key2', sample_df, ttl_hours=24)
        cache.set('key3', sample_df, ttl_hours=24)

        # Expire key2
        expires_at = datetime.now() - timedelta(hours=1)
        cache.metadata['key2']['expires_at'] = expires_at.isoformat()

        # Clear expired
        cleared = cache.clear_expired()

        assert cleared == 1
        assert cache.exists('key1')
        assert not cache.exists('key2')
        assert cache.exists('key3')

    def test_no_expiration(self, cache, sample_df):
        """Test cache entry with no expiration."""
        # Set default_ttl_hours to 0 (no expiration)
        cache.default_ttl_hours = 0
        cache.set('test_key', sample_df, ttl_hours=0)

        # Should not have expires_at in metadata
        entry = cache.metadata['test_key']
        assert 'expires_at' not in entry or entry.get('ttl_hours', 0) == 0


class TestCacheMetadata:
    """Test cache metadata functionality."""

    def test_metadata_basic_fields(self, cache, sample_df):
        """Test basic metadata fields are stored."""
        cache.set('test_key', sample_df, ttl_hours=24)

        entry = cache.metadata['test_key']

        assert 'key' in entry
        assert 'cached_at' in entry
        assert 'size_bytes' in entry
        assert 'rows' in entry
        assert 'columns' in entry
        assert entry['rows'] == len(sample_df)
        assert entry['columns'] == list(sample_df.columns)

    def test_metadata_custom_fields(self, cache, sample_df):
        """Test custom metadata storage."""
        custom_meta = {'source': 'yfinance', 'symbol': 'AAPL'}

        cache.set('test_key', sample_df, ttl_hours=24, metadata=custom_meta)

        entry = cache.metadata['test_key']

        assert 'metadata' in entry
        assert entry['metadata']['source'] == 'yfinance'
        assert entry['metadata']['symbol'] == 'AAPL'

    def test_metadata_persistence(self, temp_cache_dir, sample_df):
        """Test metadata persists across cache instances."""
        # Create cache and set data
        cache1 = DataCache(cache_dir=temp_cache_dir)
        cache1.set('test_key', sample_df, ttl_hours=24)

        # Create new cache instance (should load metadata)
        cache2 = DataCache(cache_dir=temp_cache_dir)

        assert cache2.exists('test_key')
        assert 'test_key' in cache2.metadata


class TestCacheGetOrLoad:
    """Test get_or_load functionality."""

    def test_get_or_load_cache_hit(self, cache, sample_df):
        """Test get_or_load with cache hit."""
        # Pre-populate cache
        cache.set('test_key', sample_df)

        # Loader should not be called
        loader_called = False

        def loader():
            nonlocal loader_called
            loader_called = True
            return pd.DataFrame({'col': [1, 2, 3]})

        result = cache.get_or_load('test_key', loader, ttl_hours=24)

        assert not loader_called
        pd.testing.assert_frame_equal(result, sample_df, check_freq=False)

    def test_get_or_load_cache_miss(self, cache):
        """Test get_or_load with cache miss."""
        loader_df = pd.DataFrame({'col': [1, 2, 3]})

        def loader():
            return loader_df

        result = cache.get_or_load('test_key', loader, ttl_hours=24)

        # Should return loader result
        pd.testing.assert_frame_equal(result, loader_df)

        # Should now be cached
        assert cache.exists('test_key')

    def test_get_or_load_force_refresh(self, cache, sample_df):
        """Test get_or_load with force_refresh."""
        # Pre-populate cache
        cache.set('test_key', sample_df)

        # Force refresh
        new_df = pd.DataFrame({'col': [1, 2, 3]})

        def loader():
            return new_df

        result = cache.get_or_load('test_key', loader, ttl_hours=24, force_refresh=True)

        # Should return new data from loader
        pd.testing.assert_frame_equal(result, new_df)


class TestCacheStats:
    """Test cache statistics."""

    def test_get_stats(self, cache, sample_df):
        """Test getting cache statistics."""
        cache.set('key1', sample_df)
        cache.set('key2', sample_df)

        stats = cache.get_stats()

        assert stats['total_entries'] == 2
        assert stats['total_size_bytes'] > 0
        assert stats['total_size_mb'] > 0
        assert stats['total_rows'] == len(sample_df) * 2
        assert 'cache_dir' in stats

    def test_repr(self, cache):
        """Test string representation."""
        repr_str = repr(cache)

        assert 'DataCache' in repr_str
        assert 'entries=' in repr_str
        assert 'size=' in repr_str


class TestCacheEdgeCases:
    """Test edge cases and error handling."""

    def test_cache_empty_dataframe(self, cache):
        """Test caching empty DataFrame (should be skipped)."""
        empty_df = pd.DataFrame()

        cache.set('empty_key', empty_df)

        # Should not be cached
        assert not cache.exists('empty_key')

    def test_cache_long_key(self, cache, sample_df):
        """Test caching with very long key (should be hashed)."""
        long_key = 'a' * 200

        cache.set(long_key, sample_df)

        assert cache.exists(long_key)

        retrieved = cache.get(long_key)
        assert retrieved is not None

    def test_cache_special_chars_in_key(self, cache, sample_df):
        """Test caching with special characters in key."""
        special_key = 'key/with\\special:chars*?'

        cache.set(special_key, sample_df)

        assert cache.exists(special_key)


class TestGlobalCache:
    """Test global cache instance."""

    def test_get_global_cache(self, temp_cache_dir):
        """Test getting global cache instance."""
        cache = get_cache(cache_dir=temp_cache_dir)

        assert isinstance(cache, DataCache)
