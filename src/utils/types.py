"""
Type definitions for TradingSystemStack.

Provides TypedDict and Protocol definitions for common data structures.
"""
from typing import TypedDict, Protocol, Any, Dict, List, Optional, Union
from datetime import datetime
import pandas as pd


class OHLCVData(TypedDict):
    """OHLCV candlestick data structure.

    Attributes:
        open: Opening price
        high: Highest price
        low: Lowest price
        close: Closing price
        volume: Trading volume
    """
    open: float
    high: float
    low: float
    close: float
    volume: float


class UniverseData(TypedDict):
    """Universe of symbols with their data.

    Attributes:
        symbol: Symbol ticker
        data: DataFrame with OHLCV data
        metadata: Optional metadata
    """
    symbol: str
    data: pd.DataFrame
    metadata: Optional[Dict[str, Any]]


class IndicatorParams(TypedDict, total=False):
    """Parameters for indicator calculation.

    Attributes:
        period: Lookback period
        length: Length parameter
        fast_period: Fast EMA period
        slow_period: Slow EMA period
        signal_period: Signal line period
        std_dev: Standard deviation multiplier
        multiplier: ATR multiplier
        Any other indicator-specific parameters
    """
    period: int
    length: int
    fast_period: int
    slow_period: int
    signal_period: int
    std_dev: float
    multiplier: float


class StrategyParams(TypedDict, total=False):
    """Parameters for strategy configuration.

    Attributes:
        name: Strategy name
        params: Strategy-specific parameters
        timeframe: Trading timeframe
        initial_cash: Starting capital
        fees: Trading fees as decimal
        slippage: Slippage as decimal
    """
    name: str
    params: Dict[str, Any]
    timeframe: str
    initial_cash: float
    fees: float
    slippage: float


class BacktestResult(TypedDict):
    """Backtest execution result.

    Attributes:
        total_return: Total return percentage
        sharpe_ratio: Sharpe ratio
        max_drawdown: Maximum drawdown
        win_rate: Winning trades percentage
        total_trades: Number of trades
        avg_trade: Average trade return
        equity_curve: Equity curve series
    """
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    avg_trade: float
    equity_curve: pd.Series


class ScanResult(TypedDict):
    """Scanner result for a symbol.

    Attributes:
        symbol: Symbol ticker
        matched_at: Timestamp of match
        conditions_met: Number of conditions satisfied
        score: Match score
        details: Additional details
    """
    symbol: str
    matched_at: datetime
    conditions_met: int
    score: float
    details: Dict[str, Any]


class Pattern(TypedDict):
    """Chart pattern detection result.

    Attributes:
        name: Pattern name (e.g., 'triangle', 'head_and_shoulders')
        type: Pattern type (e.g., 'bullish', 'bearish', 'neutral')
        start_idx: Starting index in data
        end_idx: Ending index in data
        confidence: Confidence score (0-1)
        breakout_target: Target price for breakout
        key_levels: Important price levels
    """
    name: str
    type: str
    start_idx: int
    end_idx: int
    confidence: float
    breakout_target: Optional[float]
    key_levels: Dict[str, float]


class Trendline(TypedDict):
    """Trendline definition.

    Attributes:
        type: 'support' or 'resistance'
        slope: Line slope
        intercept: Y-intercept
        touches: Number of price touches
        strength: Trendline strength (0-1)
        start_idx: Starting index
        end_idx: Ending index
        points: List of (index, price) tuples
    """
    type: str
    slope: float
    intercept: float
    touches: int
    strength: float
    start_idx: int
    end_idx: int
    points: List[tuple[int, float]]


class Zone(TypedDict):
    """Supply/Demand zone definition.

    Attributes:
        type: 'supply' or 'demand'
        top: Upper price boundary
        bottom: Lower price boundary
        strength: Zone strength (0-1)
        freshness: Zone freshness (0-1, higher = more recent)
        formed_at: Index where zone formed
        touches: Number of times price touched zone
        last_tested: Index of last test
    """
    type: str
    top: float
    bottom: float
    strength: float
    freshness: float
    formed_at: int
    touches: int
    last_tested: Optional[int]


class SentimentScore(TypedDict):
    """Sentiment analysis score.

    Attributes:
        timestamp: Score timestamp
        news_sentiment: News sentiment (-1 to 1)
        social_sentiment: Social media sentiment (-1 to 1)
        fear_greed: Fear & Greed index (0-100)
        composite_score: Weighted composite (-1 to 1)
        sources: Source breakdown
    """
    timestamp: datetime
    news_sentiment: float
    social_sentiment: float
    fear_greed: float
    composite_score: float
    sources: Dict[str, Any]


# Protocols for structural typing

class Indicator(Protocol):
    """Protocol for indicator implementations."""

    def calculate(self, data: pd.DataFrame, **params: Any) -> Union[pd.Series, pd.DataFrame]:
        """Calculate indicator values."""
        ...

    @property
    def name(self) -> str:
        """Indicator name."""
        ...


class Strategy(Protocol):
    """Protocol for trading strategy implementations."""

    def generate_signals(self, data: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        """Generate entry and exit signals.

        Returns:
            tuple: (entries, exits) as boolean Series
        """
        ...

    @property
    def name(self) -> str:
        """Strategy name."""
        ...

    @property
    def params(self) -> Dict[str, Any]:
        """Strategy parameters."""
        ...


class DataLoader(Protocol):
    """Protocol for data loader implementations."""

    def load(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        interval: str
    ) -> pd.DataFrame:
        """Load OHLCV data for symbol."""
        ...


# Type aliases
TimeFrame = str  # '1m', '5m', '15m', '1h', '4h', '1d', '1w', '1M'
Symbol = str
Price = float
Volume = float
Timestamp = Union[datetime, pd.Timestamp]
DateRange = tuple[datetime, datetime]
