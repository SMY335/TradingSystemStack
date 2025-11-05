"""
Pairs Trading Strategy - Statistical Arbitrage

Implements pairs trading based on cointegration analysis.
Trades the spread between two correlated assets when it deviates from mean.
"""

import backtrader as bt
import pandas as pd
import numpy as np
from numpy.linalg import LinAlgError
import logging

try:
    from statsmodels.tsa.stattools import coint
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

# Set up logging
logger = logging.getLogger(__name__)


class PairsTradingStrategy(bt.Strategy):
    """
    Pairs Trading based on cointegration
    
    - Identify cointegrated pairs (e.g., BTC/ETH)
    - Calculate spread = price_A - hedge_ratio * price_B
    - Enter when spread > threshold (mean reversion)
    - Exit when spread returns to mean
    """
    
    params = (
        ('lookback', 60),
        ('entry_zscore', 2.0),
        ('exit_zscore', 0.5),
        ('stop_loss_zscore', 3.0),
        ('position_size', 1.0),
        ('recalc_period', 20),  # Recalculate hedge ratio every N bars
    )
    
    def __init__(self):
        # Primary and secondary data feeds
        self.data_a = self.datas[0]  # First asset (e.g., BTC)
        self.data_b = self.datas[1]  # Second asset (e.g., ETH)
        
        self.hedge_ratio = None
        self.spread_history = []
        self.zscore = None
        
        self.bar_count = 0
        self.position_type = None  # 'long_spread' or 'short_spread'
        
        # Check if statsmodels is available
        if not STATSMODELS_AVAILABLE:
            print("⚠️ WARNING: statsmodels not available. Using simple linear regression for hedge ratio.")
    
    def next(self):
        self.bar_count += 1
        
        # Need enough data for lookback
        if len(self) < self.params.lookback:
            return
        
        # Recalculate hedge ratio periodically
        if self.hedge_ratio is None or self.bar_count % self.params.recalc_period == 0:
            self._calculate_hedge_ratio()
        
        # Calculate current spread and z-score
        self._calculate_spread()
        
        # Trading logic
        if not self.position:
            self._check_entry()
        else:
            self._check_exit()
    
    def _calculate_hedge_ratio(self):
        """Calculate hedge ratio using linear regression"""
        prices_a = np.array([self.data_a.close[-i] for i in range(self.params.lookback, -1, -1)])
        prices_b = np.array([self.data_b.close[-i] for i in range(self.params.lookback, -1, -1)])
        
        # Simple linear regression: hedge_ratio = cov(A,B) / var(B)
        self.hedge_ratio = np.cov(prices_a, prices_b)[0, 1] / np.var(prices_b)
        
        # Alternative: use OLS slope
        # self.hedge_ratio = np.polyfit(prices_b, prices_a, 1)[0]
        
        if STATSMODELS_AVAILABLE and len(prices_a) > 30:
            # Test for cointegration
            try:
                _, p_value, _ = coint(prices_a, prices_b)
                if p_value > 0.05:
                    logger.warning(f"Pairs not cointegrated (p-value: {p_value:.4f})")
                    print(f"⚠️ Warning: Pairs not cointegrated (p-value: {p_value:.4f})")
            except (ValueError, LinAlgError) as e:
                logger.warning(f"Failed to test cointegration: {e}. Proceeding with calculated hedge ratio.")
            except Exception as e:
                logger.error(f"Unexpected error testing cointegration: {e}", exc_info=True)
    
    def _calculate_spread(self):
        """Calculate spread and z-score"""
        # Current spread
        current_spread = self.data_a.close[0] - self.hedge_ratio * self.data_b.close[0]
        self.spread_history.append(current_spread)
        
        # Keep only lookback period
        if len(self.spread_history) > self.params.lookback:
            self.spread_history.pop(0)
        
        # Z-score
        if len(self.spread_history) >= self.params.lookback:
            mean = np.mean(self.spread_history)
            std = np.std(self.spread_history)
            
            if std > 0:
                self.zscore = (current_spread - mean) / std
            else:
                self.zscore = 0
        else:
            self.zscore = 0
    
    def _check_entry(self):
        """Check for entry signals"""
        if self.zscore is None:
            return
        
        # Spread too high: SHORT spread (short A, long B)
        if self.zscore > self.params.entry_zscore:
            self._enter_short_spread()
        
        # Spread too low: LONG spread (long A, short B)
        elif self.zscore < -self.params.entry_zscore:
            self._enter_long_spread()
    
    def _enter_long_spread(self):
        """Enter long spread position (long A, short B)"""
        self.buy(data=self.data_a, size=self.params.position_size)
        self.sell(data=self.data_b, size=self.hedge_ratio * self.params.position_size)
        self.position_type = 'long_spread'
        print(f"📈 LONG SPREAD @ z-score: {self.zscore:.2f} | Hedge: {self.hedge_ratio:.4f}")
    
    def _enter_short_spread(self):
        """Enter short spread position (short A, long B)"""
        self.sell(data=self.data_a, size=self.params.position_size)
        self.buy(data=self.data_b, size=self.hedge_ratio * self.params.position_size)
        self.position_type = 'short_spread'
        print(f"📉 SHORT SPREAD @ z-score: {self.zscore:.2f} | Hedge: {self.hedge_ratio:.4f}")
    
    def _check_exit(self):
        """Check for exit signals"""
        if self.zscore is None:
            return
        
        # Exit on mean reversion
        if abs(self.zscore) < self.params.exit_zscore:
            self._exit_position("Mean Reversion")
        
        # Stop loss on divergence
        elif abs(self.zscore) > self.params.stop_loss_zscore:
            self._exit_position("Stop Loss")
    
    def _exit_position(self, reason: str):
        """Exit all positions"""
        self.close(data=self.data_a)
        self.close(data=self.data_b)
        print(f"🔄 EXIT {self.position_type.upper()} @ z-score: {self.zscore:.2f} | Reason: {reason}")
        self.position_type = None
    
    def notify_order(self, order):
        """Handle order notifications"""
        if order.status in [order.Completed]:
            if order.isbuy():
                pass  # Buy completed
            elif order.issell():
                pass  # Sell completed
        
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            print(f"❌ Order {order.Status[order.status]}")
    
    def notify_trade(self, trade):
        """Handle trade notifications"""
        if trade.isclosed:
            pnl = trade.pnl
            if pnl > 0:
                print(f"✅ Trade Closed | PnL: ${pnl:.2f}")
            else:
                print(f"❌ Trade Closed | PnL: ${pnl:.2f}")
