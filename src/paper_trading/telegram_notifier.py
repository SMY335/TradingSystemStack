"""
Telegram notifications for paper trading bot
"""
from __future__ import annotations
import logging
from typing import Optional
import requests

logger = logging.getLogger(__name__)


class TelegramNotifier:
    """Send notifications via Telegram bot"""

    def __init__(self, bot_token: Optional[str] = None, chat_id: Optional[str] = None):
        """
        Initialize Telegram notifier

        Args:
            bot_token: Telegram bot token (from @BotFather)
            chat_id: Chat ID to send messages to

        To get your bot token:
            1. Open Telegram and search for @BotFather
            2. Send /newbot and follow instructions
            3. Copy the token provided

        To get your chat ID:
            1. Send a message to your bot
            2. Visit: https://api.telegram.org/bot<YOUR_BOT_TOKEN>/getUpdates
            3. Find your chat_id in the response
        """
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.enabled = bot_token is not None and chat_id is not None

        if not self.enabled:
            logger.warning("Telegram notifications disabled (no token/chat_id provided)")
        else:
            logger.info("Telegram notifications enabled")

    def send_message(self, message: str, parse_mode: str = "Markdown") -> bool:
        """
        Send a message via Telegram

        Args:
            message: Message text
            parse_mode: Markdown or HTML

        Returns:
            True if sent successfully
        """
        if not self.enabled:
            logger.debug(f"Telegram message (not sent): {message}")
            return False

        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            data = {
                "chat_id": self.chat_id,
                "text": message,
                "parse_mode": parse_mode
            }

            response = requests.post(url, data=data, timeout=10)
            response.raise_for_status()

            logger.debug("Telegram message sent successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")
            return False

    def notify_bot_started(self, strategy: str, symbol: str, capital: float):
        """Notify that bot has started"""
        message = (
            f"🚀 *Bot Started*\n\n"
            f"Strategy: `{strategy}`\n"
            f"Symbol: `{symbol}`\n"
            f"Capital: `${capital:,.2f}`"
        )
        self.send_message(message)

    def notify_bot_stopped(self, stats: dict):
        """Notify that bot has stopped"""
        message = (
            f"⏹️ *Bot Stopped*\n\n"
            f"Final Value: `${stats['total_value']:,.2f}`\n"
            f"Total P&L: `${stats['total_pnl']:,.2f}` ({stats['total_pnl_pct']:+.2f}%)\n"
            f"Total Trades: `{stats['num_trades']}`\n"
            f"Win Rate: `{stats['win_rate']:.2f}%`"
        )
        self.send_message(message)

    def notify_trade_opened(self, symbol: str, side: str, price: float, quantity: float):
        """Notify when a position is opened"""
        emoji = "🔵" if side.upper() == "BUY" else "🔴"
        message = (
            f"{emoji} *Position Opened*\n\n"
            f"Side: `{side.upper()}`\n"
            f"Symbol: `{symbol}`\n"
            f"Price: `${price:,.2f}`\n"
            f"Quantity: `{quantity:.6f}`\n"
            f"Value: `${price * quantity:,.2f}`"
        )
        self.send_message(message)

    def notify_trade_closed(self, symbol: str, pnl: float, pnl_pct: float, duration_hours: float):
        """Notify when a position is closed"""
        emoji = "✅" if pnl > 0 else "❌"
        message = (
            f"{emoji} *Position Closed*\n\n"
            f"Symbol: `{symbol}`\n"
            f"P&L: `${pnl:,.2f}` ({pnl_pct:+.2f}%)\n"
            f"Duration: `{duration_hours:.1f} hours`"
        )
        self.send_message(message)

    def notify_error(self, error_msg: str):
        """Notify about an error"""
        message = f"⚠️ *Error*\n\n`{error_msg}`"
        self.send_message(message)

    def notify_daily_summary(self, stats: dict):
        """Send daily performance summary"""
        message = (
            f"📊 *Daily Summary*\n\n"
            f"Portfolio Value: `${stats['total_value']:,.2f}`\n"
            f"Today's P&L: `${stats['total_pnl']:,.2f}` ({stats['total_pnl_pct']:+.2f}%)\n"
            f"Trades Today: `{stats['num_trades']}`\n"
            f"Win Rate: `{stats['win_rate']:.2f}%`\n"
            f"Open Positions: `{stats['num_open_positions']}`"
        )
        self.send_message(message)


# Example usage and setup instructions
if __name__ == "__main__":
    print("="*60)
    print("TELEGRAM BOT SETUP INSTRUCTIONS")
    print("="*60)
    print()
    print("1. Create a Telegram Bot:")
    print("   - Open Telegram and search for @BotFather")
    print("   - Send: /newbot")
    print("   - Follow instructions to create your bot")
    print("   - Copy the bot token provided")
    print()
    print("2. Get Your Chat ID:")
    print("   - Send any message to your bot")
    print("   - Visit: https://api.telegram.org/bot<YOUR_BOT_TOKEN>/getUpdates")
    print("   - Find 'chat':{'id': YOUR_CHAT_ID} in the response")
    print()
    print("3. Test Your Setup:")
    print("   - Run this script with your credentials:")
    print()
    print("   from src.paper_trading.telegram_notifier import TelegramNotifier")
    print()
    print("   notifier = TelegramNotifier(")
    print("       bot_token='YOUR_BOT_TOKEN',")
    print("       chat_id='YOUR_CHAT_ID'")
    print("   )")
    print("   notifier.send_message('Test message!')")
    print()
    print("="*60)
