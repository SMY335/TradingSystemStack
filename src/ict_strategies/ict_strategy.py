"""
Complete ICT Strategy - Combining Order Blocks, FVG, and Liquidity Pools

This strategy implements the full Inner Circle Trader methodology for
institutional-style trading.
"""

import backtrader as bt
import pandas as pd
from src.ict_strategies.order_blocks import OrderBlockDetector, OrderBlockType
from src.ict_strategies.fair_value_gaps import FairValueGapDetector, FVGType
from src.ict_strategies.liquidity_pools import LiquidityPoolDetector


class ICTStrategy(bt.Strategy):
    """
    Complete ICT Strategy
    
    Logic:
    1. Identify liquidity pools (resistances/supports)
    2. Wait for liquidity sweep (breakout + rejection)
    3. Look for order block in opposite direction
    4. Enter on FVG fill or order block retest
    5. TP = opposite liquidity pool, SL = behind order block
    """
    
    params = (
        ('order_block_lookback', 20),
        ('fvg_min_gap_pct', 0.001),
        ('liquidity_lookback', 50),
        ('risk_reward_ratio', 2.0),
        ('position_size_pct', 0.02),
        ('max_positions', 1),
    )
    
    def __init__(self):
        self.ob_detector = OrderBlockDetector(lookback=self.params.order_block_lookback)
        self.fvg_detector = FairValueGapDetector(min_gap_pct=self.params.fvg_min_gap_pct)
        self.liq_detector = LiquidityPoolDetector(lookback=self.params.liquidity_lookback)
        
        self.order_blocks = []
        self.fvgs = []
        self.buy_pools = []
        self.sell_pools = []
        
        self.order = None
        self.entry_price = None
        self.stop_loss = None
        self.take_profit = None
    
    def next(self):
        # Convert Backtrader data to DataFrame
        df = self._get_dataframe()
        
        if len(df) < self.params.liquidity_lookback + 10:
            return
        
        # Detect ICT structures
        self.order_blocks = self.ob_detector.detect(df)
        self.fvgs = self.fvg_detector.detect(df)
        self.buy_pools, self.sell_pools = self.liq_detector.detect_pools(df)
        
        # Update FVGs
        self.fvgs = self.fvg_detector.update_gaps(self.fvgs, df, len(df)-1)
        
        # If position open, manage it
        if self.position:
            self._manage_position()
            return
        
        # Check for entry setup
        if len(self.order_blocks) > 0 and (len(self.buy_pools) > 0 or len(self.sell_pools) > 0):
            self._check_for_entry(df)
    
    def _check_for_entry(self, df: pd.DataFrame):
        """Look for a valid ICT setup"""
        current_price = self.data.close[0]
        
        # LONG Setup
        # 1. Liquidity sweep below (price touches sell-side pool and rejects)
        # 2. Bullish order block present
        # 3. Price near order block
        
        for pool in self.sell_pools[-5:]:  # Check recent pools
            if self._is_liquidity_swept(pool, df, 'sell'):
                # Look for bullish OB
                bullish_obs = [ob for ob in self.order_blocks 
                              if ob.type == OrderBlockType.BULLISH and not ob.mitigated
                              and ob.start_idx > len(df) - 50]  # Recent OBs
                
                if bullish_obs:
                    best_ob = max(bullish_obs, key=lambda x: x.strength)
                    
                    # Price must be close to the OB
                    if abs(current_price - best_ob.low) / current_price < 0.005:  # Within 0.5%
                        self._enter_long(best_ob, pool)
                        return
        
        # SHORT Setup (inverse logic)
        for pool in self.buy_pools[-5:]:
            if self._is_liquidity_swept(pool, df, 'buy'):
                bearish_obs = [ob for ob in self.order_blocks 
                              if ob.type == OrderBlockType.BEARISH and not ob.mitigated
                              and ob.start_idx > len(df) - 50]
                
                if bearish_obs:
                    best_ob = max(bearish_obs, key=lambda x: x.strength)
                    
                    if abs(current_price - best_ob.high) / current_price < 0.005:
                        self._enter_short(best_ob, pool)
                        return
    
    def _is_liquidity_swept(self, pool, df: pd.DataFrame, pool_type: str) -> bool:
        """Check if liquidity was swept recently (last 3 candles)"""
        if len(df) < 3:
            return False
        
        recent = df.iloc[-3:]
        
        if pool_type == 'sell':
            # Price must break below and reject
            return any(recent['low'] < pool.price_level) and recent.iloc[-1]['close'] > pool.price_level
        else:
            # Price must break above and reject
            return any(recent['high'] > pool.price_level) and recent.iloc[-1]['close'] < pool.price_level
    
    def _enter_long(self, order_block, target_pool):
        """Enter long position"""
        self.entry_price = self.data.close[0]
        self.stop_loss = order_block.low * 0.995  # 0.5% below OB
        
        # Find TP = next buy-side pool
        target_pools = [p for p in self.buy_pools if p.price_level > self.entry_price]
        if target_pools:
            self.take_profit = min(target_pools, key=lambda p: p.price_level).price_level
        else:
            # Use risk-reward ratio
            risk = self.entry_price - self.stop_loss
            self.take_profit = self.entry_price + (risk * self.params.risk_reward_ratio)
        
        size = self._calculate_position_size(self.entry_price, self.stop_loss)
        if size > 0:
            self.order = self.buy(size=size)
            print(f"🟢 LONG @ {self.entry_price:.2f} | SL: {self.stop_loss:.2f} | TP: {self.take_profit:.2f}")
    
    def _enter_short(self, order_block, target_pool):
        """Enter short position"""
        self.entry_price = self.data.close[0]
        self.stop_loss = order_block.high * 1.005  # 0.5% above OB
        
        # Find TP = next sell-side pool
        target_pools = [p for p in self.sell_pools if p.price_level < self.entry_price]
        if target_pools:
            self.take_profit = max(target_pools, key=lambda p: p.price_level).price_level
        else:
            risk = self.stop_loss - self.entry_price
            self.take_profit = self.entry_price - (risk * self.params.risk_reward_ratio)
        
        size = self._calculate_position_size(self.entry_price, self.stop_loss)
        if size > 0:
            self.order = self.sell(size=size)
            print(f"🔴 SHORT @ {self.entry_price:.2f} | SL: {self.stop_loss:.2f} | TP: {self.take_profit:.2f}")
    
    def _calculate_position_size(self, entry: float, stop_loss: float) -> float:
        """Position sizing based on risk"""
        account_value = self.broker.getvalue()
        risk_amount = account_value * self.params.position_size_pct
        risk_per_unit = abs(entry - stop_loss)
        
        if risk_per_unit == 0:
            return 0
        
        return risk_amount / risk_per_unit
    
    def _manage_position(self):
        """Manage open position with SL/TP"""
        current_price = self.data.close[0]
        
        if self.position.size > 0:  # Long position
            if current_price <= self.stop_loss:
                self.close()
                print(f"❌ LONG SL Hit @ {current_price:.2f}")
            elif current_price >= self.take_profit:
                self.close()
                print(f"✅ LONG TP Hit @ {current_price:.2f}")
        
        elif self.position.size < 0:  # Short position
            if current_price >= self.stop_loss:
                self.close()
                print(f"❌ SHORT SL Hit @ {current_price:.2f}")
            elif current_price <= self.take_profit:
                self.close()
                print(f"✅ SHORT TP Hit @ {current_price:.2f}")
    
    def _get_dataframe(self) -> pd.DataFrame:
        """Convert Backtrader data to DataFrame"""
        lookback = min(200, len(self.data))
        data = {
            'open': [self.data.open[-i] for i in range(lookback, -1, -1)],
            'high': [self.data.high[-i] for i in range(lookback, -1, -1)],
            'low': [self.data.low[-i] for i in range(lookback, -1, -1)],
            'close': [self.data.close[-i] for i in range(lookback, -1, -1)],
            'volume': [self.data.volume[-i] for i in range(lookback, -1, -1)],
        }
        return pd.DataFrame(data)
