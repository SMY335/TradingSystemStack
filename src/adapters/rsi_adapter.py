"""
RSI Strategy Adapter
Adapts RSI strategy for Nautilus Trader and Backtrader
"""
from typing import Dict
from src.adapters.base_strategy_adapter import BaseStrategyAdapter, StrategyConfig

# Nautilus imports
from nautilus_trader.trading.strategy import Strategy
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.orders import MarketOrder

# Technical analysis
import talib
import numpy as np

# Backtrader imports
import backtrader as bt


class NautilusRSIStrategy(Strategy):
    """RSI Strategy for Nautilus Trader"""
    
    def __init__(self, config):
        super().__init__(config)
        self.period = config.parameters['period']
        self.oversold = config.parameters['oversold']
        self.overbought = config.parameters['overbought']
        self.instrument_id = None
        self.bar_type = None
        self.prices = []  # Store price history for talib
        self.position_size = config.capital * config.risk_params['max_position_size']
        
    def on_start(self):
        """Called when strategy starts"""
        # Get first symbol from config
        symbol = self.config.symbols[0].replace('/', '')  # BTC/USDT -> BTCUSDT
        self.instrument_id = InstrumentId.from_str(f"{symbol}.BINANCE")
        
        # Create bar type for subscription
        self.bar_type = BarType.from_str(
            f"{symbol}.BINANCE-{self.config.timeframe}-LAST"
        )
        
        # Subscribe to bars
        self.subscribe_bars(self.bar_type)
        
        self.log.info(
            f"RSI Strategy started: period={self.period}, "
            f"oversold={self.oversold}, overbought={self.overbought}"
        )
    
    def on_bar(self, bar: Bar):
        """Called on each bar update"""
        # Collect price history
        self.prices.append(bar.close.as_double())
        
        # Wait for enough data
        if len(self.prices) < self.period:
            return
        
        # Calculate RSI using talib
        prices_array = np.array(self.prices[-self.period*2:])  # Need extra data for RSI calculation
        rsi = talib.RSI(prices_array, timeperiod=self.period)[-1]
        
        # Get current position
        position = self.cache.position(self.instrument_id)
        
        # Trading logic: Buy when RSI < oversold, Sell when RSI > overbought
        if rsi < self.oversold and (position is None or position.is_closed):
            # Buy signal - oversold
            self.log.info(f"BUY signal: RSI={rsi:.2f} < {self.oversold}")
            self._enter_long()
            
        elif rsi > self.overbought and position and position.is_long:
            # Sell signal - overbought
            self.log.info(f"SELL signal: RSI={rsi:.2f} > {self.overbought}")
            self._exit_position()
    
    def _enter_long(self):
        """Enter long position"""
        order = self.order_factory.market(
            instrument_id=self.instrument_id,
            order_side=OrderSide.BUY,
            quantity=self.position_size
        )
        self.submit_order(order)
    
    def _exit_position(self):
        """Exit current position"""
        position = self.cache.position(self.instrument_id)
        if position:
            order = self.order_factory.market(
                instrument_id=self.instrument_id,
                order_side=OrderSide.SELL,
                quantity=position.quantity
            )
            self.submit_order(order)
    
    def on_stop(self):
        """Called when strategy stops"""
        self.log.info("RSI Strategy stopped")


class BacktraderRSIStrategy(bt.Strategy):
    """RSI Strategy for Backtrader"""
    
    params = (
        ('period', 14),
        ('oversold', 30),
        ('overbought', 70),
        ('max_position_size', 0.1),
        ('stop_loss_pct', 0.02),
    )
    
    def __init__(self):
        """Initialize strategy"""
        # Validate period
        if not isinstance(self.params.period, int):
            raise TypeError(f"period must be int, got {type(self.params.period)}")
        
        if self.params.period <= 0 or self.params.period > 100:
            raise ValueError(f"period must be between 1 and 100, got {self.params.period}")
        
        # Validate oversold
        if not isinstance(self.params.oversold, int):
            raise TypeError(f"oversold must be int, got {type(self.params.oversold)}")
        
        if self.params.oversold <= 0 or self.params.oversold >= 50:
            raise ValueError(f"oversold must be between 1 and 49, got {self.params.oversold}")
        
        # Validate overbought
        if not isinstance(self.params.overbought, int):
            raise TypeError(f"overbought must be int, got {type(self.params.overbought)}")
        
        if self.params.overbought <= 50 or self.params.overbought >= 100:
            raise ValueError(f"overbought must be between 51 and 99, got {self.params.overbought}")
        
        # Validate relationship
        if self.params.oversold >= self.params.overbought:
            raise ValueError(
                f"oversold ({self.params.oversold}) must be < overbought ({self.params.overbought})"
            )
        
        # Create RSI indicator
        self.rsi = bt.indicators.RelativeStrengthIndex(
            self.data.close,
            period=self.params.period
        )
        
        # Track orders
        self.order = None
        
    def log(self, txt, dt=None):
        """Logging function"""
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()} {txt}')
    
    def notify_order(self, order):
        """Called when order status changes"""
        if order.status in [order.Submitted, order.Accepted]:
            return
        
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED, Price: {order.executed.price:.2f}')
            elif order.issell():
                self.log(f'SELL EXECUTED, Price: {order.executed.price:.2f}')
        
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')
        
        self.order = None
    
    def notify_trade(self, trade):
        """Called when trade is closed"""
        if trade.isclosed:
            self.log(f'TRADE PROFIT, Gross: {trade.pnl:.2f}, Net: {trade.pnlcomm:.2f}')
    
    def next(self):
        """Called on each bar"""
        # Check if an order is pending
        if self.order:
            return
        
        # Skip if RSI is not yet calculated
        if len(self.rsi) < 1:
            return
        
        rsi_value = self.rsi[0]
        
        # Check if we are in the market
        if not self.position:
            # Not in market, check for buy signal (oversold)
            if rsi_value < self.params.oversold:
                # Calculate position size
                cash = self.broker.getcash()
                size = (cash * self.params.max_position_size) / self.data.close[0]
                self.log(f'BUY CREATE, RSI: {rsi_value:.2f}, Price: {self.data.close[0]:.2f}')
                self.order = self.buy(size=size)
        
        else:
            # In market, check for sell signal (overbought)
            if rsi_value > self.params.overbought:
                self.log(f'SELL CREATE, RSI: {rsi_value:.2f}, Price: {self.data.close[0]:.2f}')
                self.order = self.sell(size=self.position.size)


class RSIAdapter(BaseStrategyAdapter):
    """Adapter for RSI Strategy"""
    
    def validate_parameters(self) -> None:
        """Validate RSI strategy parameters"""
        required = ['period', 'oversold', 'overbought']
        for param in required:
            if param not in self.config.parameters:
                raise ValueError(f"Missing required parameter: {param}")
        
        period = self.config.parameters['period']
        oversold = self.config.parameters['oversold']
        overbought = self.config.parameters['overbought']
        
        if period < 2:
            raise ValueError("Period must be at least 2")
        
        if not (0 < oversold < overbought < 100):
            raise ValueError(
                f"Must satisfy: 0 < oversold ({oversold}) < overbought ({overbought}) < 100"
            )
    
    def to_nautilus(self):
        """Convert to Nautilus Trader strategy"""
        return NautilusRSIStrategy(self.config)
    
    def to_backtrader(self):
        """Convert to Backtrader strategy class"""
        # Create a configured strategy class
        class ConfiguredRSIStrategy(BacktraderRSIStrategy):
            params = (
                ('period', self.config.parameters['period']),
                ('oversold', self.config.parameters['oversold']),
                ('overbought', self.config.parameters['overbought']),
                ('max_position_size', self.config.risk_params['max_position_size']),
                ('stop_loss_pct', self.config.risk_params['stop_loss_pct']),
            )
        
        return ConfiguredRSIStrategy
    
    def get_parameter_space(self) -> Dict[str, tuple]:
        """Get optimization parameter space"""
        return {
            'period': (7, 28),
            'oversold': (20, 40),
            'overbought': (60, 80)
        }
