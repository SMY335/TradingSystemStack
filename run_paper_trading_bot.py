#!/usr/bin/env python3
"""
Launch Paper Trading Bot - CLI Mode

This script runs the bot in the terminal with detailed logging.
For GUI mode, use: streamlit run src/dashboard/live_dashboard.py
"""
import argparse
from src.strategies import AVAILABLE_STRATEGIES, EMAStrategy
from src.paper_trading import LiveTradingBot
from src.paper_trading.logger_config import setup_logger
from src.paper_trading.telegram_notifier import TelegramNotifier


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Launch Paper Trading Bot")

    parser.add_argument(
        '--strategy',
        type=str,
        default='EMA Crossover',
        choices=list(AVAILABLE_STRATEGIES.keys()),
        help='Trading strategy to use'
    )

    parser.add_argument(
        '--symbol',
        type=str,
        default='BTC/USDT',
        help='Trading pair (e.g., BTC/USDT, ETH/USDT)'
    )

    parser.add_argument(
        '--timeframe',
        type=str,
        default='1h',
        choices=['1m', '5m', '15m', '1h', '4h', '1d'],
        help='Candlestick timeframe'
    )

    parser.add_argument(
        '--exchange',
        type=str,
        default='binance',
        choices=['binance', 'kraken', 'coinbase', 'bybit', 'okx'],
        help='Exchange to connect to'
    )

    parser.add_argument(
        '--capital',
        type=float,
        default=10000,
        help='Initial capital in USD'
    )

    parser.add_argument(
        '--fees',
        type=float,
        default=0.1,
        help='Trading fees percentage (0.1 = 0.1%%)'
    )

    parser.add_argument(
        '--slippage',
        type=float,
        default=0.05,
        help='Slippage percentage (0.05 = 0.05%%)'
    )

    parser.add_argument(
        '--interval',
        type=int,
        default=60,
        help='Check interval in seconds'
    )

    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )

    parser.add_argument(
        '--telegram-token',
        type=str,
        default=None,
        help='Telegram bot token for notifications'
    )

    parser.add_argument(
        '--telegram-chat-id',
        type=str,
        default=None,
        help='Telegram chat ID for notifications'
    )

    # Strategy-specific parameters
    parser.add_argument('--ema-fast', type=int, default=12, help='EMA fast period')
    parser.add_argument('--ema-slow', type=int, default=26, help='EMA slow period')
    parser.add_argument('--rsi-period', type=int, default=14, help='RSI period')
    parser.add_argument('--rsi-oversold', type=int, default=30, help='RSI oversold level')
    parser.add_argument('--rsi-overbought', type=int, default=70, help='RSI overbought level')
    parser.add_argument('--macd-fast', type=int, default=12, help='MACD fast period')
    parser.add_argument('--macd-slow', type=int, default=26, help='MACD slow period')
    parser.add_argument('--macd-signal', type=int, default=9, help='MACD signal period')

    args = parser.parse_args()

    # Setup logging
    logger = setup_logger(log_level=args.log_level)

    # Setup Telegram notifications (optional)
    notifier = TelegramNotifier(
        bot_token=args.telegram_token,
        chat_id=args.telegram_chat_id
    )

    # Create strategy instance with parameters
    strategy_class = AVAILABLE_STRATEGIES[args.strategy]

    if args.strategy == 'EMA Crossover':
        strategy = strategy_class(fast_period=args.ema_fast, slow_period=args.ema_slow)
    elif args.strategy == 'RSI':
        strategy = strategy_class(
            period=args.rsi_period,
            oversold=args.rsi_oversold,
            overbought=args.rsi_overbought
        )
    elif args.strategy == 'MACD':
        strategy = strategy_class(
            fast_period=args.macd_fast,
            slow_period=args.macd_slow,
            signal_period=args.macd_signal
        )
    else:
        strategy = strategy_class()

    # Display configuration
    logger.info("="*60)
    logger.info("PAPER TRADING BOT CONFIGURATION")
    logger.info("="*60)
    logger.info(f"Strategy:       {strategy.name}")
    logger.info(f"Description:    {strategy.get_description()}")
    logger.info(f"Symbol:         {args.symbol}")
    logger.info(f"Timeframe:      {args.timeframe}")
    logger.info(f"Exchange:       {args.exchange}")
    logger.info(f"Initial Capital: ${args.capital:,.2f}")
    logger.info(f"Fees:           {args.fees}%")
    logger.info(f"Slippage:       {args.slippage}%")
    logger.info(f"Check Interval: {args.interval}s")
    logger.info(f"Telegram:       {'Enabled' if notifier.enabled else 'Disabled'}")
    logger.info("="*60)

    # Create and run bot
    try:
        bot = LiveTradingBot(
            strategy=strategy,
            symbol=args.symbol,
            timeframe=args.timeframe,
            exchange_id=args.exchange,
            initial_capital=args.capital,
            fees_pct=args.fees,
            slippage_pct=args.slippage,
            check_interval=args.interval
        )

        # Send start notification
        notifier.notify_bot_started(strategy.name, args.symbol, args.capital)

        logger.info("Starting bot... (Press Ctrl+C to stop)")
        logger.info("")

        # Run bot
        bot.run()

    except KeyboardInterrupt:
        logger.info("\nBot stopped by user")

        # Send stop notification
        if bot:
            stats = bot.get_stats()
            notifier.notify_bot_stopped(stats)

    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        notifier.notify_error(str(e))


if __name__ == "__main__":
    main()
