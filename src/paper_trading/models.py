"""
Data models for paper trading
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal
from enum import Enum


class OrderSide(str, Enum):
    """Order side enum"""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(str, Enum):
    """Order status enum"""
    PENDING = "pending"
    FILLED = "filled"
    CANCELLED = "cancelled"
    FAILED = "failed"


class PositionSide(str, Enum):
    """Position side enum"""
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


@dataclass
class Order:
    """Represents a trading order"""
    id: str
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    timestamp: datetime
    status: OrderStatus = OrderStatus.PENDING
    filled_price: float = 0.0
    filled_quantity: float = 0.0
    filled_timestamp: datetime | None = None
    fee: float = 0.0
    slippage: float = 0.0

    def __post_init__(self):
        if isinstance(self.side, str):
            self.side = OrderSide(self.side)
        if isinstance(self.status, str):
            self.status = OrderStatus(self.status)


@dataclass
class Position:
    """Represents an open position"""
    symbol: str
    side: PositionSide
    entry_price: float
    quantity: float
    entry_timestamp: datetime
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0

    def update_price(self, current_price: float):
        """Update current price and calculate unrealized P&L"""
        self.current_price = current_price

        if self.side == PositionSide.LONG:
            self.unrealized_pnl = (current_price - self.entry_price) * self.quantity
            self.unrealized_pnl_pct = ((current_price - self.entry_price) / self.entry_price) * 100
        elif self.side == PositionSide.SHORT:
            self.unrealized_pnl = (self.entry_price - current_price) * self.quantity
            self.unrealized_pnl_pct = ((self.entry_price - current_price) / self.entry_price) * 100

    def __post_init__(self):
        if isinstance(self.side, str):
            self.side = PositionSide(self.side)


@dataclass
class Trade:
    """Represents a completed trade"""
    id: str
    symbol: str
    side: OrderSide
    entry_price: float
    exit_price: float
    quantity: float
    entry_timestamp: datetime
    exit_timestamp: datetime
    pnl: float = 0.0
    pnl_pct: float = 0.0
    fees: float = 0.0
    duration_seconds: float = 0.0

    def __post_init__(self):
        if isinstance(self.side, str):
            self.side = OrderSide(self.side)

        # Calculate P&L
        if self.side == OrderSide.BUY:
            self.pnl = (self.exit_price - self.entry_price) * self.quantity - self.fees
            self.pnl_pct = ((self.exit_price - self.entry_price) / self.entry_price) * 100
        else:
            self.pnl = (self.entry_price - self.exit_price) * self.quantity - self.fees
            self.pnl_pct = ((self.entry_price - self.exit_price) / self.entry_price) * 100

        # Calculate duration
        self.duration_seconds = (self.exit_timestamp - self.entry_timestamp).total_seconds()

    @property
    def is_winning(self) -> bool:
        """Check if trade was profitable"""
        return self.pnl > 0


@dataclass
class Portfolio:
    """Represents the trading portfolio"""
    initial_capital: float
    current_capital: float
    positions: list[Position] = field(default_factory=list)
    trades: list[Trade] = field(default_factory=list)
    total_fees: float = 0.0

    @property
    def total_value(self) -> float:
        """Total portfolio value including open positions"""
        unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions)
        return self.current_capital + unrealized_pnl

    @property
    def total_pnl(self) -> float:
        """Total profit/loss"""
        return self.total_value - self.initial_capital

    @property
    def total_pnl_pct(self) -> float:
        """Total profit/loss percentage"""
        return (self.total_pnl / self.initial_capital) * 100

    @property
    def num_trades(self) -> int:
        """Total number of completed trades"""
        return len(self.trades)

    @property
    def num_winning_trades(self) -> int:
        """Number of winning trades"""
        return sum(1 for t in self.trades if t.is_winning)

    @property
    def win_rate(self) -> float:
        """Win rate percentage"""
        if self.num_trades == 0:
            return 0.0
        return (self.num_winning_trades / self.num_trades) * 100

    @property
    def num_open_positions(self) -> int:
        """Number of open positions"""
        return len(self.positions)
