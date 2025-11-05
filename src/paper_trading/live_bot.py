"""
Live Paper Trading Bot - Runs strategies in real-time
"""
from __future__ import annotations
import time
import logging
from datetime import datetime, timedelta
from typing import Optional
import pandas as pd
import ccxt
from threading import Thread, Event

from .engine import PaperTradingEngine
from .models import OrderSide, PositionSide
from ..strategies.base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


class LiveTradingBot:
    """Bot that executes strategies in real-time using paper trading"""

    def __init__(
        self,
        strategy: BaseStrategy,
        symbol: str,
        timeframe: str = '1h',
        exchange_id: str = 'binance',
        initial_capital: float = 10000,
        fees_pct: float = 0.1,
        slippage_pct: float = 0.05,
        check_interval: int = 60
    ):
        """
        Initialize live trading bot

        Args:
            strategy: Trading strategy to execute
            symbol: Trading pair (e.g., 'BTC/USDT')
            timeframe: Candle timeframe (e.g., '1h', '15m')
            exchange_id: Exchange to connect to
            initial_capital: Starting capital
            fees_pct: Trading fees percentage
            slippage_pct: Slippage percentage
            check_interval: How often to check for signals (seconds)
        """
        self.strategy = strategy
        self.symbol = symbol
        self.timeframe = timeframe
        self.check_interval = check_interval

        # Initialize exchange
        self.exchange = getattr(ccxt, exchange_id)({
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'}
        })

        # Initialize paper trading engine
        self.engine = PaperTradingEngine(
            initial_capital=initial_capital,
            fees_pct=fees_pct,
            slippage_pct=slippage_pct
        )

        # State
        self.is_running = False
        self.stop_event = Event()
        self.last_check_time = None
        self.historical_data = pd.DataFrame()

        # Statistics
        self.total_checks = 0
        self.total_signals = 0

        logger.info(f"Live Trading Bot initialized: {strategy.name} on {symbol}")

    def fetch_historical_data(self, lookback_periods: int = 200) -> pd.DataFrame:
        """
        Fetch historical data for strategy calculation

        Args:
            lookback_periods: Number of candles to fetch

        Returns:
            DataFrame with OHLCV data
        """
        try:
            ohlcv = self.exchange.fetch_ohlcv(
                self.symbol,
                timeframe=self.timeframe,
                limit=lookback_periods
            )

            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)

            return df

        except Exception as e:
            logger.error(f"Error fetching historical data: {e}")
            return pd.DataFrame()

    def get_current_price(self) -> Optional[float]:
        """Get current market price"""
        try:
            ticker = self.exchange.fetch_ticker(self.symbol)
            return ticker['last']
        except Exception as e:
            logger.error(f"Error fetching current price: {e}")
            return None

    def check_signals(self):
        """Check for trading signals and execute if needed"""
        try:
            self.total_checks += 1

            # Fetch latest data
            logger.debug(f"Fetching data for {self.symbol}...")
            df = self.fetch_historical_data()

            if df.empty:
                logger.warning("No data received")
                return

            self.historical_data = df

            # Get current price
            current_price = df['close'].iloc[-1]
            self.engine.update_price(self.symbol, current_price)

            # Generate signals
            entries, exits = self.strategy.generate_signals(df)

            # Get last signal
            last_entry = entries.iloc[-1] if len(entries) > 0 else False
            last_exit = exits.iloc[-1] if len(exits) > 0 else False

            # Check current position
            position = self.engine.get_position(self.symbol)

            # Execute trades based on signals
            if last_entry and position is None:
                # Entry signal and no position
                logger.info(f"🔵 BUY signal detected at ${current_price:,.2f}")
                self.engine.place_order(
                    symbol=self.symbol,
                    side=OrderSide.BUY,
                    price=current_price
                )
                self.total_signals += 1

            elif last_exit and position is not None:
                # Exit signal and have position
                logger.info(f"🔴 SELL signal detected at ${current_price:,.2f}")
                self.engine.place_order(
                    symbol=self.symbol,
                    side=OrderSide.SELL,
                    quantity=position.quantity,
                    price=current_price
                )
                self.total_signals += 1

            # Log status
            stats = self.engine.get_stats()
            logger.info(
                f"Status: ${stats['total_value']:,.2f} | "
                f"P&L: {stats['total_pnl_pct']:+.2f}% | "
                f"Trades: {stats['num_trades']} | "
                f"Positions: {stats['num_open_positions']}"
            )

        except Exception as e:
            logger.error(f"Error checking signals: {e}", exc_info=True)

    def run_once(self):
        """Run one iteration (for testing)"""
        self.check_signals()

    def run(self):
        """Start the bot (runs indefinitely)"""
        self.is_running = True
        self.stop_event.clear()

        logger.info(f"🚀 Bot starting: {self.strategy.name} on {self.symbol}")
        logger.info(f"Check interval: {self.check_interval}s")

        try:
            while not self.stop_event.is_set():
                self.last_check_time = datetime.now()
                self.check_signals()

                # Wait until next check
                self.stop_event.wait(self.check_interval)

        except KeyboardInterrupt:
            logger.info("Bot stopped by user")
        except Exception as e:
            logger.error(f"Bot error: {e}", exc_info=True)
        finally:
            self.stop()

    def run_async(self):
        """Run bot in background thread"""
        thread = Thread(target=self.run, daemon=True)
        thread.start()
        logger.info("Bot running in background")
        return thread

    def stop(self):
        """Stop the bot"""
        logger.info("Stopping bot...")
        self.is_running = False
        self.stop_event.set()

        # Close all positions
        self.engine.close_all_positions()

        # Final stats
        stats = self.engine.get_stats()
        logger.info("="*60)
        logger.info("FINAL STATISTICS")
        logger.info("="*60)
        logger.info(f"Initial Capital:    ${self.engine.initial_capital:,.2f}")
        logger.info(f"Final Value:        ${stats['total_value']:,.2f}")
        logger.info(f"Total P&L:          ${stats['total_pnl']:,.2f} ({stats['total_pnl_pct']:+.2f}%)")
        logger.info(f"Total Trades:       {stats['num_trades']}")
        logger.info(f"Win Rate:           {stats['win_rate']:.2f}%")
        logger.info(f"Total Fees:         ${stats['total_fees']:,.2f}")
        logger.info(f"Total Checks:       {self.total_checks}")
        logger.info(f"Total Signals:      {self.total_signals}")
        logger.info("="*60)

    def get_status(self) -> dict:
        """Get current bot status"""
        stats = self.engine.get_stats()

        return {
            'is_running': self.is_running,
            'strategy': self.strategy.name,
            'symbol': self.symbol,
            'last_check': self.last_check_time,
            'total_checks': self.total_checks,
            'total_signals': self.total_signals,
            **stats
        }
