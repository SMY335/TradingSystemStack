"""
FRED (Federal Reserve Economic Data) integration for TradingSystemStack.

Provides access to 800,000+ economic time series from the St. Louis Fed.
"""
from datetime import datetime, timedelta
from typing import Optional, Union, List, Dict
import pandas as pd
import logging
import os

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

logger = logging.getLogger(__name__)


# Popular FRED series IDs
POPULAR_SERIES = {
    # GDP & Growth
    'gdp': 'GDP',
    'gdp_growth': 'A191RL1Q225SBEA',
    'real_gdp': 'GDPC1',

    # Employment
    'unemployment': 'UNRATE',
    'nonfarm_payrolls': 'PAYEMS',
    'labor_force': 'CIVPART',
    'initial_claims': 'ICSA',

    # Inflation
    'cpi': 'CPIAUCSL',
    'core_cpi': 'CPILFESL',
    'pce': 'PCE',
    'core_pce': 'PCEPILFE',

    # Interest Rates
    'fed_funds': 'FEDFUNDS',
    'treasury_10y': 'DGS10',
    'treasury_2y': 'DGS2',
    'treasury_3m': 'DGS3MO',

    # Housing
    'housing_starts': 'HOUST',
    'home_sales': 'HSN1F',
    'case_shiller': 'CSUSHPISA',

    # Manufacturing
    'industrial_production': 'INDPRO',
    'capacity_utilization': 'TCU',
    'pmi': 'MANEMP',

    # Retail & Consumer
    'retail_sales': 'RSXFS',
    'consumer_sentiment': 'UMCSENT',
    'personal_income': 'PI',
    'personal_spending': 'PCE',

    # Money Supply
    'm1': 'M1SL',
    'm2': 'M2SL',

    # Markets
    'sp500': 'SP500',
    'vix': 'VIXCLS',
    'ted_spread': 'TEDRATE',
}


class FREDError(Exception):
    """Custom exception for FRED operations."""
    pass


class FREDClient:
    """Client for FRED API.

    Requires API key from https://fred.stlouisfed.org/docs/api/api_key.html

    Examples:
        >>> client = FREDClient(api_key='your_api_key')
        >>> df = client.get_series('GDP')
        >>> df = client.get_series('unemployment', start='2020-01-01')
    """

    BASE_URL = 'https://api.stlouisfed.org/fred'

    def __init__(self, api_key: Optional[str] = None):
        """Initialize FRED client.

        Args:
            api_key: FRED API key (or set FRED_API_KEY env var)
        """
        if not REQUESTS_AVAILABLE:
            raise FREDError(
                "requests not available. Install: pip install requests"
            )

        self.api_key = api_key or os.getenv('FRED_API_KEY')
        if not self.api_key:
            raise FREDError(
                "FRED API key required. Get one at: "
                "https://fred.stlouisfed.org/docs/api/api_key.html"
            )

        self.session = requests.Session()
        self.session.params = {'api_key': self.api_key, 'file_type': 'json'}

    def get_series(
        self,
        series_id: str,
        start: Optional[Union[str, datetime]] = None,
        end: Optional[Union[str, datetime]] = None,
        frequency: Optional[str] = None,
        aggregation: Optional[str] = None
    ) -> pd.DataFrame:
        """Get time series data from FRED.

        Args:
            series_id: FRED series ID (e.g., 'GDP', 'UNRATE')
            start: Start date (default: earliest available)
            end: End date (default: latest available)
            frequency: Resample frequency ('d', 'w', 'm', 'q', 'a')
            aggregation: Aggregation method ('avg', 'sum', 'eop')

        Returns:
            DataFrame with DatetimeIndex and 'value' column

        Raises:
            FREDError: If API request fails

        Examples:
            >>> client = FREDClient()
            >>> df = client.get_series('GDP')
            >>> df = client.get_series('UNRATE', start='2020-01-01')
        """
        # Convert dates to strings
        params = {}
        if start:
            if isinstance(start, datetime):
                start = start.strftime('%Y-%m-%d')
            params['observation_start'] = start

        if end:
            if isinstance(end, datetime):
                end = end.strftime('%Y-%m-%d')
            params['observation_end'] = end

        if frequency:
            params['frequency'] = frequency

        if aggregation:
            params['aggregation_method'] = aggregation

        # Make API request
        url = f"{self.BASE_URL}/series/observations"
        params['series_id'] = series_id

        logger.debug(f"Fetching FRED series: {series_id}")

        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"FRED API request failed: {e}")
            raise FREDError(f"Failed to fetch {series_id}: {e}") from e

        # Parse observations
        if 'observations' not in data:
            raise FREDError(f"No data returned for {series_id}")

        observations = data['observations']

        if not observations:
            raise FREDError(f"No observations for {series_id}")

        # Convert to DataFrame
        df = pd.DataFrame(observations)

        # Convert date and value
        df['date'] = pd.to_datetime(df['date'])
        df['value'] = pd.to_numeric(df['value'], errors='coerce')

        # Set index
        df = df.set_index('date')
        df = df[['value']]  # Keep only value column

        # Remove NaN values
        df = df.dropna()

        # Ensure sorted
        df = df.sort_index()

        logger.info(
            f"Fetched {len(df)} observations for {series_id} "
            f"({df.index[0]} to {df.index[-1]})"
        )

        return df

    def get_series_info(self, series_id: str) -> Dict[str, any]:
        """Get metadata about a series.

        Args:
            series_id: FRED series ID

        Returns:
            Dictionary with series metadata

        Examples:
            >>> client = FREDClient()
            >>> info = client.get_series_info('GDP')
            >>> print(info['title'])
            'Gross Domestic Product'
        """
        url = f"{self.BASE_URL}/series"
        params = {'series_id': series_id}

        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"FRED API request failed: {e}")
            raise FREDError(f"Failed to fetch info for {series_id}: {e}") from e

        if 'seriess' not in data or not data['seriess']:
            raise FREDError(f"No info found for {series_id}")

        return data['seriess'][0]

    def search_series(
        self,
        search_text: str,
        limit: int = 10
    ) -> List[Dict[str, any]]:
        """Search for series by text.

        Args:
            search_text: Search query
            limit: Maximum results to return

        Returns:
            List of series metadata dictionaries

        Examples:
            >>> client = FREDClient()
            >>> results = client.search_series('unemployment')
        """
        url = f"{self.BASE_URL}/series/search"
        params = {'search_text': search_text, 'limit': limit}

        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"FRED API request failed: {e}")
            raise FREDError(f"Search failed: {e}") from e

        if 'seriess' not in data:
            return []

        return data['seriess']


def get_series(
    series_id: str,
    start: Optional[Union[str, datetime]] = None,
    end: Optional[Union[str, datetime]] = None,
    api_key: Optional[str] = None
) -> pd.DataFrame:
    """Get FRED time series data (convenience function).

    Args:
        series_id: FRED series ID or alias (e.g., 'GDP', 'unemployment')
        start: Start date
        end: End date
        api_key: FRED API key (or use FRED_API_KEY env var)

    Returns:
        DataFrame with DatetimeIndex and 'value' column

    Examples:
        >>> # Using series ID
        >>> df = get_series('GDP')

        >>> # Using alias
        >>> df = get_series('unemployment', start='2020-01-01')

        >>> # Get multiple series
        >>> gdp = get_series('gdp')
        >>> unemployment = get_series('unemployment')
    """
    # Resolve alias to series ID
    series_id_resolved = POPULAR_SERIES.get(series_id.lower(), series_id)

    # Create client and fetch
    client = FREDClient(api_key=api_key)
    return client.get_series(series_id_resolved, start=start, end=end)


def get_multiple_series(
    series_ids: List[str],
    start: Optional[Union[str, datetime]] = None,
    end: Optional[Union[str, datetime]] = None,
    api_key: Optional[str] = None
) -> pd.DataFrame:
    """Get multiple FRED series and merge into single DataFrame.

    Args:
        series_ids: List of series IDs or aliases
        start: Start date
        end: End date
        api_key: FRED API key

    Returns:
        DataFrame with DatetimeIndex and columns for each series

    Examples:
        >>> df = get_multiple_series(['gdp', 'unemployment', 'cpi'])
        >>> df = get_multiple_series(['GDP', 'UNRATE'], start='2020-01-01')
    """
    client = FREDClient(api_key=api_key)

    dfs = {}
    for series_id in series_ids:
        # Resolve alias
        series_id_resolved = POPULAR_SERIES.get(series_id.lower(), series_id)

        try:
            df = client.get_series(series_id_resolved, start=start, end=end)
            dfs[series_id] = df['value']
        except FREDError as e:
            logger.warning(f"Failed to fetch {series_id}: {e}")
            continue

    if not dfs:
        raise FREDError("No series data retrieved")

    # Merge all series
    result = pd.DataFrame(dfs)

    logger.info(
        f"Retrieved {len(result.columns)} series with {len(result)} observations"
    )

    return result


def get_popular_series() -> Dict[str, str]:
    """Get dictionary of popular series aliases and their IDs.

    Returns:
        Dictionary mapping aliases to FRED series IDs

    Examples:
        >>> series = get_popular_series()
        >>> series['unemployment']
        'UNRATE'
    """
    return POPULAR_SERIES.copy()


def is_fred_available() -> bool:
    """Check if FRED API is available.

    Returns:
        True if requests library installed and API key configured
    """
    if not REQUESTS_AVAILABLE:
        return False

    api_key = os.getenv('FRED_API_KEY')
    return api_key is not None
