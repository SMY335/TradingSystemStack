"""
Data models for portfolio management
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional
from enum import Enum


class AssetType(Enum):
    """Type of asset"""
    CRYPTO = "crypto"
    STOCK = "stock"
    FOREX = "forex"
    COMMODITY = "commodity"
    BOND = "bond"


@dataclass
class Asset:
    """Represents a tradable asset"""
    symbol: str
    asset_type: AssetType
    exchange: Optional[str] = None

    def __str__(self):
        return f"{self.symbol} ({self.asset_type.value})"


@dataclass
class Position:
    """Represents a position in an asset"""
    asset: Asset
    quantity: float
    entry_price: float
    entry_date: datetime
    current_price: Optional[float] = None

    @property
    def market_value(self) -> float:
        """Current market value of the position"""
        price = self.current_price if self.current_price else self.entry_price
        return self.quantity * price

    @property
    def unrealized_pnl(self) -> float:
        """Unrealized profit/loss"""
        if self.current_price is None:
            return 0.0
        return (self.current_price - self.entry_price) * self.quantity

    @property
    def unrealized_pnl_pct(self) -> float:
        """Unrealized profit/loss percentage"""
        if self.current_price is None:
            return 0.0
        return ((self.current_price - self.entry_price) / self.entry_price) * 100


@dataclass
class Portfolio:
    """Represents a multi-asset portfolio"""
    name: str
    initial_capital: float
    cash: float
    positions: List[Position] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)

    @property
    def total_market_value(self) -> float:
        """Total market value of all positions"""
        return sum(pos.market_value for pos in self.positions)

    @property
    def total_value(self) -> float:
        """Total portfolio value (cash + positions)"""
        return self.cash + self.total_market_value

    @property
    def total_pnl(self) -> float:
        """Total unrealized profit/loss"""
        return sum(pos.unrealized_pnl for pos in self.positions)

    @property
    def total_return_pct(self) -> float:
        """Total return percentage since inception"""
        if self.initial_capital == 0:
            return 0.0
        return ((self.total_value - self.initial_capital) / self.initial_capital) * 100

    @property
    def weights(self) -> Dict[str, float]:
        """Current portfolio weights"""
        total = self.total_value
        if total == 0:
            return {}

        weights = {}
        for pos in self.positions:
            weights[pos.asset.symbol] = pos.market_value / total
        weights['CASH'] = self.cash / total
        return weights

    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for a specific symbol"""
        for pos in self.positions:
            if pos.asset.symbol == symbol:
                return pos
        return None

    def add_position(self, position: Position) -> None:
        """Add a new position"""
        self.positions.append(position)

    def remove_position(self, symbol: str) -> bool:
        """Remove a position"""
        for i, pos in enumerate(self.positions):
            if pos.asset.symbol == symbol:
                self.positions.pop(i)
                return True
        return False

    def update_prices(self, prices: Dict[str, float]) -> None:
        """Update current prices for all positions"""
        for pos in self.positions:
            if pos.asset.symbol in prices:
                pos.current_price = prices[pos.asset.symbol]

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            'name': self.name,
            'initial_capital': self.initial_capital,
            'cash': self.cash,
            'total_value': self.total_value,
            'total_pnl': self.total_pnl,
            'total_return_pct': self.total_return_pct,
            'weights': self.weights,
            'positions': [
                {
                    'symbol': pos.asset.symbol,
                    'quantity': pos.quantity,
                    'entry_price': pos.entry_price,
                    'current_price': pos.current_price,
                    'market_value': pos.market_value,
                    'unrealized_pnl': pos.unrealized_pnl,
                    'unrealized_pnl_pct': pos.unrealized_pnl_pct
                }
                for pos in self.positions
            ]
        }
