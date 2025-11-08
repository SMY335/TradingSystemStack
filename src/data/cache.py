"""
Intelligent data caching system for TradingSystemStack.

Provides parquet-based caching with TTL, automatic invalidation,
and efficient storage management.
"""
from datetime import datetime, timedelta
from typing import Optional, Union, Dict, List, Callable, Any
from pathlib import Path
import pandas as pd
import hashlib
import json
import logging

from src.utils.io import read_parquet, write_parquet, ensure_directory

logger = logging.getLogger(__name__)


class CacheError(Exception):
    """Custom exception for cache operations."""
    pass


class DataCache:
    """Intelligent data cache with TTL and automatic invalidation.

    Examples:
        >>> cache = DataCache(cache_dir='data/cache')
        >>> cache.set('AAPL_1d', df, ttl_hours=24)
        >>> df = cache.get('AAPL_1d')
        >>> cache.clear('AAPL_1d')
    """

    METADATA_FILE = '_cache_metadata.json'

    def __init__(
        self,
        cache_dir: Union[str, Path] = 'data/cache',
        default_ttl_hours: int = 24,
        compression: str = 'snappy'
    ):
        """Initialize cache.

        Args:
            cache_dir: Directory for cache files
            default_ttl_hours: Default TTL in hours
            compression: Parquet compression ('snappy', 'gzip', 'brotli')
        """
        self.cache_dir = Path(cache_dir)
        self.default_ttl_hours = default_ttl_hours
        self.compression = compression

        # Create cache directory
        ensure_directory(self.cache_dir)

        # Load metadata
        self.metadata_path = self.cache_dir / self.METADATA_FILE
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> Dict[str, Any]:
        """Load cache metadata from JSON file."""
        if self.metadata_path.exists():
            try:
                with open(self.metadata_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache metadata: {e}")
                return {}
        return {}

    def _save_metadata(self) -> None:
        """Save cache metadata to JSON file."""
        try:
            with open(self.metadata_path, 'w') as f:
                json.dump(self.metadata, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save cache metadata: {e}")

    def _get_cache_key(self, key: str) -> str:
        """Generate cache key (hash if needed).

        Args:
            key: Original key

        Returns:
            Safe filename
        """
        # Hash long keys
        if len(key) > 100 or any(c in key for c in ['/', '\\', ':', '*', '?', '"', '<', '>', '|']):
            return hashlib.md5(key.encode()).hexdigest()
        return key

    def _get_cache_path(self, key: str) -> Path:
        """Get cache file path for key."""
        cache_key = self._get_cache_key(key)
        return self.cache_dir / f"{cache_key}.parquet"

    def _is_expired(self, key: str) -> bool:
        """Check if cache entry is expired."""
        if key not in self.metadata:
            return True

        entry = self.metadata[key]
        if 'expires_at' not in entry:
            return False

        expires_at = datetime.fromisoformat(entry['expires_at'])
        return datetime.now() >= expires_at

    def set(
        self,
        key: str,
        data: pd.DataFrame,
        ttl_hours: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Store data in cache.

        Args:
            key: Cache key
            data: DataFrame to cache
            ttl_hours: TTL in hours (None = no expiration)
            metadata: Additional metadata to store

        Examples:
            >>> cache.set('AAPL_1d', df, ttl_hours=24)
            >>> cache.set('BTC_1h', df, ttl_hours=1, metadata={'source': 'binance'})
        """
        if data.empty:
            logger.warning(f"Not caching empty DataFrame: {key}")
            return

        # Get cache path
        cache_path = self._get_cache_path(key)

        # Write parquet
        try:
            write_parquet(data, cache_path, compression=self.compression)
        except Exception as e:
            logger.error(f"Failed to write cache: {e}")
            raise CacheError(f"Failed to cache {key}: {e}") from e

        # Update metadata
        entry = {
            'key': key,
            'cached_at': datetime.now().isoformat(),
            'size_bytes': cache_path.stat().st_size,
            'rows': len(data),
            'columns': list(data.columns),
        }

        # Add expiration
        if ttl_hours is None:
            ttl_hours = self.default_ttl_hours

        if ttl_hours > 0:
            expires_at = datetime.now() + timedelta(hours=ttl_hours)
            entry['expires_at'] = expires_at.isoformat()
            entry['ttl_hours'] = ttl_hours

        # Add custom metadata
        if metadata:
            entry['metadata'] = metadata

        self.metadata[key] = entry
        self._save_metadata()

        logger.debug(f"Cached {key} ({len(data)} rows, {ttl_hours}h TTL)")

    def get(
        self,
        key: str,
        default: Optional[pd.DataFrame] = None
    ) -> Optional[pd.DataFrame]:
        """Get data from cache.

        Args:
            key: Cache key
            default: Default value if not found or expired

        Returns:
            Cached DataFrame or default

        Examples:
            >>> df = cache.get('AAPL_1d')
            >>> df = cache.get('AAPL_1d', default=pd.DataFrame())
        """
        # Check if expired
        if self._is_expired(key):
            logger.debug(f"Cache expired: {key}")
            self.clear(key)
            return default

        # Get cache path
        cache_path = self._get_cache_path(key)

        if not cache_path.exists():
            logger.debug(f"Cache miss: {key}")
            return default

        # Read parquet
        try:
            df = read_parquet(cache_path)
            logger.debug(f"Cache hit: {key} ({len(df)} rows)")
            return df
        except Exception as e:
            logger.error(f"Failed to read cache: {e}")
            return default

    def exists(self, key: str) -> bool:
        """Check if key exists and is not expired.

        Args:
            key: Cache key

        Returns:
            True if exists and valid

        Examples:
            >>> if cache.exists('AAPL_1d'):
            ...     df = cache.get('AAPL_1d')
        """
        if self._is_expired(key):
            return False

        cache_path = self._get_cache_path(key)
        return cache_path.exists()

    def clear(self, key: Optional[str] = None) -> None:
        """Clear cache entry or entire cache.

        Args:
            key: Cache key (None = clear all)

        Examples:
            >>> cache.clear('AAPL_1d')  # Clear one entry
            >>> cache.clear()  # Clear all
        """
        if key is None:
            # Clear all cache
            logger.info("Clearing entire cache")

            # Delete all parquet files
            for file in self.cache_dir.glob("*.parquet"):
                try:
                    file.unlink()
                except Exception as e:
                    logger.warning(f"Failed to delete {file}: {e}")

            # Clear metadata
            self.metadata = {}
            self._save_metadata()

        else:
            # Clear specific entry
            cache_path = self._get_cache_path(key)

            if cache_path.exists():
                try:
                    cache_path.unlink()
                    logger.debug(f"Deleted cache file: {key}")
                except Exception as e:
                    logger.warning(f"Failed to delete cache file: {e}")

            # Remove from metadata
            if key in self.metadata:
                del self.metadata[key]
                self._save_metadata()

    def clear_expired(self) -> int:
        """Clear all expired cache entries.

        Returns:
            Number of entries cleared

        Examples:
            >>> cleared = cache.clear_expired()
            >>> print(f"Cleared {cleared} expired entries")
        """
        expired_keys = [
            key for key in self.metadata.keys()
            if self._is_expired(key)
        ]

        for key in expired_keys:
            self.clear(key)

        if expired_keys:
            logger.info(f"Cleared {len(expired_keys)} expired entries")

        return len(expired_keys)

    def get_or_load(
        self,
        key: str,
        loader: Callable[[], pd.DataFrame],
        ttl_hours: Optional[int] = None,
        force_refresh: bool = False
    ) -> pd.DataFrame:
        """Get from cache or load using loader function.

        Args:
            key: Cache key
            loader: Function that loads data if not cached
            ttl_hours: TTL in hours
            force_refresh: Force reload even if cached

        Returns:
            DataFrame

        Examples:
            >>> df = cache.get_or_load(
            ...     'AAPL_1d',
            ...     lambda: yf.download('AAPL'),
            ...     ttl_hours=24
            ... )
        """
        # Check cache first
        if not force_refresh:
            cached = self.get(key)
            if cached is not None:
                return cached

        # Load data
        logger.debug(f"Loading data for {key}")
        data = loader()

        # Cache result
        if not data.empty:
            self.set(key, data, ttl_hours=ttl_hours)

        return data

    def list_keys(self) -> List[str]:
        """List all cache keys.

        Returns:
            List of cache keys

        Examples:
            >>> keys = cache.list_keys()
            >>> print(keys)
            ['AAPL_1d', 'BTC_1h', 'ETH_1h']
        """
        return list(self.metadata.keys())

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache stats

        Examples:
            >>> stats = cache.get_stats()
            >>> print(f"Cache size: {stats['total_size_mb']:.2f} MB")
        """
        total_size = sum(
            entry.get('size_bytes', 0)
            for entry in self.metadata.values()
        )

        total_rows = sum(
            entry.get('rows', 0)
            for entry in self.metadata.values()
        )

        expired = sum(
            1 for key in self.metadata.keys()
            if self._is_expired(key)
        )

        return {
            'total_entries': len(self.metadata),
            'total_size_bytes': total_size,
            'total_size_mb': total_size / (1024 * 1024),
            'total_rows': total_rows,
            'expired_entries': expired,
            'cache_dir': str(self.cache_dir),
        }

    def __len__(self) -> int:
        """Get number of cache entries."""
        return len(self.metadata)

    def __contains__(self, key: str) -> bool:
        """Check if key exists in cache."""
        return self.exists(key)

    def __repr__(self) -> str:
        """String representation."""
        stats = self.get_stats()
        return (
            f"DataCache(entries={stats['total_entries']}, "
            f"size={stats['total_size_mb']:.2f}MB, "
            f"dir={self.cache_dir})"
        )


# Global cache instance
_global_cache: Optional[DataCache] = None


def get_cache(
    cache_dir: Union[str, Path] = 'data/cache',
    default_ttl_hours: int = 24
) -> DataCache:
    """Get global cache instance.

    Args:
        cache_dir: Cache directory
        default_ttl_hours: Default TTL

    Returns:
        DataCache instance

    Examples:
        >>> cache = get_cache()
        >>> cache.set('key', df)
    """
    global _global_cache

    if _global_cache is None:
        _global_cache = DataCache(
            cache_dir=cache_dir,
            default_ttl_hours=default_ttl_hours
        )

    return _global_cache
