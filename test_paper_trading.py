"""
Test Paper Trading System
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import time
from src.strategies import EMAStrategy
from src.paper_trading import LiveTradingBot
from src.paper_trading.logger_config import setup_logger

# Setup logging
logger = setup_logger(log_level="INFO")

print("="*60)
print("🧪 TESTING PAPER TRADING SYSTEM")
print("="*60)
print()

# Test 1: Create bot
print("1️⃣ Creating paper trading bot...")
try:
    strategy = EMAStrategy(fast_period=12, slow_period=26)
    bot = LiveTradingBot(
        strategy=strategy,
        symbol='BTC/USDT',
        timeframe='1h',
        exchange_id='binance',
        initial_capital=10000,
        check_interval=10  # Check every 10 seconds for testing
    )
    print(f"   ✓ Bot created: {strategy.name} on BTC/USDT")
except Exception as e:
    print(f"   ✗ Error: {e}")
    sys.exit(1)

# Test 2: Run one iteration
print("\n2️⃣ Running one check cycle...")
try:
    bot.run_once()
    print("   ✓ Check cycle completed")
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Check status
print("\n3️⃣ Checking bot status...")
try:
    status = bot.get_status()
    print(f"   ✓ Strategy: {status['strategy']}")
    print(f"   ✓ Symbol: {status['symbol']}")
    print(f"   ✓ Portfolio Value: ${status['total_value']:,.2f}")
    print(f"   ✓ Total Checks: {status['total_checks']}")
    print(f"   ✓ Open Positions: {status['num_open_positions']}")
except Exception as e:
    print(f"   ✗ Error: {e}")
    sys.exit(1)

# Test 4: Test paper trading engine directly
print("\n4️⃣ Testing paper trading engine...")
try:
    from src.paper_trading import PaperTradingEngine, OrderSide

    engine = PaperTradingEngine(initial_capital=10000)

    # Simulate a price
    engine.update_price('BTC/USDT', 50000)

    # Place buy order
    buy_order = engine.place_order('BTC/USDT', OrderSide.BUY, price=50000)
    print(f"   ✓ Buy order placed: {buy_order.filled_quantity:.6f} BTC")

    # Update price
    engine.update_price('BTC/USDT', 52000)

    # Check unrealized P&L
    position = engine.get_position('BTC/USDT')
    if position:
        print(f"   ✓ Position: {position.quantity:.6f} BTC")
        print(f"   ✓ Entry: ${position.entry_price:,.2f}")
        print(f"   ✓ Current: ${position.current_price:,.2f}")
        print(f"   ✓ Unrealized P&L: ${position.unrealized_pnl:,.2f} ({position.unrealized_pnl_pct:+.2f}%)")

    # Sell
    sell_order = engine.place_order('BTC/USDT', OrderSide.SELL, quantity=position.quantity, price=52000)
    print(f"   ✓ Sell order placed")

    # Check stats
    stats = engine.get_stats()
    print(f"   ✓ Total P&L: ${stats['total_pnl']:,.2f} ({stats['total_pnl_pct']:+.2f}%)")
    print(f"   ✓ Trades completed: {stats['num_trades']}")

except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*60)
print("✅ ALL TESTS PASSED!")
print("="*60)
print()
print("🚀 Your paper trading system is ready!")
print()
print("Next steps:")
print("  1. Run live bot in terminal:")
print("     python run_paper_trading_bot.py --symbol BTC/USDT --timeframe 1h")
print()
print("  2. Or launch the live dashboard:")
print("     ./run_live_dashboard.sh")
print()
print("  3. Monitor logs in the 'logs/' directory")
print()
print("⚠️  IMPORTANT: This is PAPER TRADING (no real money)")
print("    Test thoroughly before considering live trading!")
