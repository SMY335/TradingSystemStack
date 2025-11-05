"""
Simple Market Maker Strategy

Implements a basic market making strategy with bid-ask spread
and inventory management.
"""

import backtrader as bt


class SimpleMarketMaker(bt.Strategy):
    """
    Simple Market Maker
    
    - Place bid/ask around mid price
    - Spread = commission * 2 + profit margin
    - Inventory management (prevent excessive one-sided positions)
    """
    
    params = (
        ('spread_bps', 20),  # 20 basis points (0.20%)
        ('max_inventory', 10),  # Maximum position size
        ('order_size', 1.0),
        ('skew_factor', 0.001),  # Price skew per inventory unit
        ('update_freq', 1),  # Update orders every N bars
    )
    
    def __init__(self):
        self.inventory = 0
        self.bid_order = None
        self.ask_order = None
        
        self.bar_count = 0
        self.total_trades = 0
        self.profitable_trades = 0
    
    def next(self):
        self.bar_count += 1
        
        # Update orders every N bars
        if self.bar_count % self.params.update_freq != 0:
            return
        
        mid_price = self.data.close[0]
        
        # Cancel existing orders
        if self.bid_order:
            self.cancel(self.bid_order)
            self.bid_order = None
        if self.ask_order:
            self.cancel(self.ask_order)
            self.ask_order = None
        
        # Calculate bid/ask prices
        spread = mid_price * (self.params.spread_bps / 10000)
        bid_price = mid_price - spread / 2
        ask_price = mid_price + spread / 2
        
        # Apply inventory skew
        # When inventory is positive (long), decrease bid and ask to encourage selling
        # When inventory is negative (short), increase bid and ask to encourage buying
        inventory_skew = self.inventory / self.params.max_inventory * self.params.skew_factor
        bid_price *= (1 - inventory_skew)
        ask_price *= (1 + inventory_skew)
        
        # Place new orders based on inventory limits
        if self.inventory < self.params.max_inventory:
            # Can buy more
            self.bid_order = self.buy(
                price=bid_price,
                exectype=bt.Order.Limit,
                size=self.params.order_size
            )
        
        if self.inventory > -self.params.max_inventory:
            # Can sell more
            self.ask_order = self.sell(
                price=ask_price,
                exectype=bt.Order.Limit,
                size=self.params.order_size
            )
    
    def notify_order(self, order):
        """Handle order notifications"""
        if order.status in [order.Completed]:
            if order.isbuy():
                self.inventory += order.executed.size
                print(f"✅ BID FILLED @ {order.executed.price:.2f} | Inventory: {self.inventory:.2f}")
            elif order.issell():
                self.inventory -= order.executed.size
                print(f"✅ ASK FILLED @ {order.executed.price:.2f} | Inventory: {self.inventory:.2f}")
        
        elif order.status in [order.Canceled, order.Expired]:
            pass  # Order canceled or expired
        
        elif order.status in [order.Margin, order.Rejected]:
            print(f"❌ Order Rejected: {order.Status[order.status]}")
    
    def notify_trade(self, trade):
        """Handle trade notifications"""
        if trade.isclosed:
            self.total_trades += 1
            pnl = trade.pnl
            
            if pnl > 0:
                self.profitable_trades += 1
                print(f"💰 Trade Closed | PnL: ${pnl:.2f} | Win Rate: {self.profitable_trades/self.total_trades*100:.1f}%")
            else:
                print(f"📉 Trade Closed | PnL: ${pnl:.2f} | Win Rate: {self.profitable_trades/self.total_trades*100:.1f}%")
    
    def stop(self):
        """Called when strategy stops"""
        print(f"\n📊 Market Making Summary:")
        print(f"   Total Trades: {self.total_trades}")
        if self.total_trades > 0:
            print(f"   Win Rate: {self.profitable_trades/self.total_trades*100:.1f}%")
        print(f"   Final Inventory: {self.inventory:.2f}")
        print(f"   Final Value: ${self.broker.getvalue():.2f}")
