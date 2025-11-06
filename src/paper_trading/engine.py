"""
Paper Trading Engine - Simulates live trading without real money
"""
from __future__ import annotations
import uuid
from datetime import datetime
from typing import Optional
import logging

from .models import (
    Order, Position, Trade, Portfolio,
    OrderSide, OrderStatus, PositionSide
)

logger = logging.getLogger(__name__)


class PaperTradingEngine:
    """Paper trading engine that simulates order execution"""

    def __init__(
        self,
        initial_capital: float = 10000,
        fees_pct: float = 0.1,
        slippage_pct: float = 0.05,
        max_position_size_pct: float = 100.0
    ):
        """
        Initialize paper trading engine

        Args:
            initial_capital: Starting capital
            fees_pct: Trading fees percentage (0.1 = 0.1%)
            slippage_pct: Slippage percentage (0.05 = 0.05%)
            max_position_size_pct: Max position size as % of capital (100 = 100%)
        """
        # Validation
        if not isinstance(initial_capital, (int, float)):
            raise TypeError(f"initial_capital must be numeric, got {type(initial_capital).__name__}")
        if initial_capital <= 0:
            raise ValueError(f"initial_capital must be positive, got {initial_capital}")

        if not isinstance(fees_pct, (int, float)):
            raise TypeError(f"fees_pct must be numeric, got {type(fees_pct).__name__}")
        if fees_pct < 0 or fees_pct > 100:
            raise ValueError(f"fees_pct must be between 0 and 100, got {fees_pct}")

        if not isinstance(slippage_pct, (int, float)):
            raise TypeError(f"slippage_pct must be numeric, got {type(slippage_pct).__name__}")
        if slippage_pct < 0 or slippage_pct > 100:
            raise ValueError(f"slippage_pct must be between 0 and 100, got {slippage_pct}")

        if not isinstance(max_position_size_pct, (int, float)):
            raise TypeError(f"max_position_size_pct must be numeric, got {type(max_position_size_pct).__name__}")
        if max_position_size_pct <= 0 or max_position_size_pct > 100:
            raise ValueError(f"max_position_size_pct must be between 0 and 100, got {max_position_size_pct}")

        self.initial_capital = initial_capital
        self.fees_pct = fees_pct / 100
        self.slippage_pct = slippage_pct / 100
        self.max_position_size_pct = max_position_size_pct / 100

        self.portfolio = Portfolio(
            initial_capital=initial_capital,
            current_capital=initial_capital
        )

        self.orders: list[Order] = []
        self.current_prices: dict[str, float] = {}

        logger.info(f"Paper Trading Engine initialized with ${initial_capital:,.2f}")

    def update_price(self, symbol: str, price: float):
        """
        Update current price for a symbol

        Args:
            symbol: Trading pair symbol
            price: Current price
        """
        self.current_prices[symbol] = price

        # Update all open positions
        for position in self.portfolio.positions:
            if position.symbol == symbol:
                position.update_price(price)

    def get_position(self, symbol: str) -> Optional[Position]:
        """Get current position for symbol"""
        for position in self.portfolio.positions:
            if position.symbol == symbol:
                return position
        return None

    def calculate_position_size(self, symbol: str, price: float) -> float:
        """
        Calculate position size based on available capital

        Args:
            symbol: Trading pair
            price: Current price

        Returns:
            Quantity to buy/sell
        """
        # Validation
        if not isinstance(symbol, str) or not symbol:
            raise ValueError(f"symbol must be non-empty string, got {symbol!r}")

        if not isinstance(price, (int, float)):
            raise TypeError(f"price must be numeric, got {type(price).__name__}")
        if price <= 0:
            raise ValueError(f"price must be positive, got {price}")

        available_capital = self.portfolio.current_capital * self.max_position_size_pct
        max_quantity = available_capital / price

        # Account for fees
        max_quantity = max_quantity * (1 - self.fees_pct)

        return max_quantity

    def place_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: Optional[float] = None,
        price: Optional[float] = None
    ) -> Order:
        """
        Place a paper trading order

        Args:
            symbol: Trading pair
            side: BUY or SELL
            quantity: Order quantity (None = calculate based on capital)
            price: Order price (None = use current market price)

        Returns:
            Order object
        """
        # Get current price
        if price is None:
            price = self.current_prices.get(symbol)
            if price is None:
                raise ValueError(f"No price available for {symbol}")

        # Calculate quantity if not provided
        if quantity is None:
            quantity = self.calculate_position_size(symbol, price)

        # Create order
        order = Order(
            id=str(uuid.uuid4())[:8],
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            timestamp=datetime.now()
        )

        self.orders.append(order)
        logger.info(f"Order placed: {side.value.upper()} {quantity:.6f} {symbol} @ ${price:,.2f}")

        # Execute immediately (market order simulation)
        self._execute_order(order)

        return order

    def _execute_order(self, order: Order):
        """
        Execute a paper trading order

        Args:
            order: Order to execute
        """
        # Simulate slippage
        if order.side == OrderSide.BUY:
            execution_price = order.price * (1 + self.slippage_pct)
        else:
            execution_price = order.price * (1 - self.slippage_pct)

        # Calculate fees
        fee = order.quantity * execution_price * self.fees_pct

        # Update order
        order.filled_price = execution_price
        order.filled_quantity = order.quantity
        order.filled_timestamp = datetime.now()
        order.fee = fee
        order.slippage = abs(execution_price - order.price) * order.quantity
        order.status = OrderStatus.FILLED

        # Update portfolio
        self._update_portfolio(order)

        logger.info(
            f"Order executed: {order.side.value.upper()} {order.filled_quantity:.6f} "
            f"{order.symbol} @ ${order.filled_price:,.2f} (fee: ${fee:.2f})"
        )

    def _update_portfolio(self, order: Order):
        """Update portfolio after order execution"""
        position = self.get_position(order.symbol)

        if order.side == OrderSide.BUY:
            # Opening or adding to long position
            if position is None:
                # New position
                new_position = Position(
                    symbol=order.symbol,
                    side=PositionSide.LONG,
                    entry_price=order.filled_price,
                    quantity=order.filled_quantity,
                    entry_timestamp=order.filled_timestamp,
                    current_price=order.filled_price
                )
                self.portfolio.positions.append(new_position)
                logger.info(f"Opened LONG position: {order.filled_quantity:.6f} {order.symbol}")

            else:
                # Adding to existing position (average price)
                total_cost = (position.entry_price * position.quantity +
                             order.filled_price * order.filled_quantity)
                total_quantity = position.quantity + order.filled_quantity
                position.entry_price = total_cost / total_quantity
                position.quantity = total_quantity
                logger.info(f"Added to LONG position: {total_quantity:.6f} {order.symbol}")

            # Deduct cost from capital
            cost = order.filled_price * order.filled_quantity + order.fee
            self.portfolio.current_capital -= cost

        else:  # SELL
            if position is None:
                logger.warning(f"Trying to sell {order.symbol} with no open position")
                return

            # Close position
            trade = Trade(
                id=str(uuid.uuid4())[:8],
                symbol=order.symbol,
                side=OrderSide.BUY,  # Original position was long
                entry_price=position.entry_price,
                exit_price=order.filled_price,
                quantity=order.filled_quantity,
                entry_timestamp=position.entry_timestamp,
                exit_timestamp=order.filled_timestamp,
                fees=order.fee + self.portfolio.total_fees
            )

            self.portfolio.trades.append(trade)
            logger.info(
                f"Closed position: {order.symbol} | P&L: ${trade.pnl:,.2f} ({trade.pnl_pct:+.2f}%)"
            )

            # Add proceeds to capital
            proceeds = order.filled_price * order.filled_quantity - order.fee
            self.portfolio.current_capital += proceeds

            # Remove position
            self.portfolio.positions.remove(position)

        # Track total fees
        self.portfolio.total_fees += order.fee

    def close_all_positions(self):
        """Close all open positions at current market prices"""
        positions_to_close = self.portfolio.positions.copy()

        for position in positions_to_close:
            current_price = self.current_prices.get(position.symbol)
            if current_price:
                self.place_order(
                    symbol=position.symbol,
                    side=OrderSide.SELL,
                    quantity=position.quantity,
                    price=current_price
                )

        logger.info("All positions closed")

    def get_stats(self) -> dict:
        """Get current portfolio statistics"""
        return {
            'total_value': self.portfolio.total_value,
            'current_capital': self.portfolio.current_capital,
            'total_pnl': self.portfolio.total_pnl,
            'total_pnl_pct': self.portfolio.total_pnl_pct,
            'num_trades': self.portfolio.num_trades,
            'win_rate': self.portfolio.win_rate,
            'num_open_positions': self.portfolio.num_open_positions,
            'total_fees': self.portfolio.total_fees
        }
