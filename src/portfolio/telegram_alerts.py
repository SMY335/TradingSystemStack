"""
Telegram Alert System for Risk Management

Sends automated alerts for critical risk events via Telegram.
"""

import asyncio
from typing import Dict, List, Optional
from telegram import Bot
from telegram.error import TelegramError
import logging
from datetime import datetime
import os

logger = logging.getLogger(__name__)


class TelegramAlerter:
    """
    Telegram alert system for risk management
    
    Features:
    - Send critical risk alerts
    - VaR breach notifications
    - Drawdown warnings
    - Correlation anomaly alerts
    - Custom formatted messages
    """
    
    def __init__(
        self,
        bot_token: Optional[str] = None,
        chat_id: Optional[str] = None
    ):
        """
        Initialize Telegram alerter
        
        Args:
            bot_token: Telegram bot token (if None, reads from env var TELEGRAM_BOT_TOKEN)
            chat_id: Chat ID to send messages to (if None, reads from env var TELEGRAM_CHAT_ID)
        """
        self.bot_token = bot_token or os.getenv('TELEGRAM_BOT_TOKEN')
        self.chat_id = chat_id or os.getenv('TELEGRAM_CHAT_ID')
        
        if not self.bot_token or not self.chat_id:
            logger.warning(
                "Telegram credentials not configured. "
                "Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID environment variables."
            )
            self.enabled = False
        else:
            self.bot = Bot(token=self.bot_token)
            self.enabled = True
    
    async def send_alert(
        self,
        title: str,
        message: str,
        alert_level: str = 'INFO'
    ) -> bool:
        """
        Send an alert message via Telegram
        
        Args:
            title: Alert title
            message: Alert message content
            alert_level: 'CRITICAL', 'WARNING', or 'INFO'
            
        Returns:
            True if sent successfully, False otherwise
        """
        if not self.enabled:
            logger.warning(f"Telegram not enabled. Would send: [{alert_level}] {title}")
            return False
        
        # Format message
        emoji_map = {
            'CRITICAL': '🚨',
            'WARNING': '⚠️',
            'INFO': 'ℹ️'
        }
        
        emoji = emoji_map.get(alert_level, 'ℹ️')
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        formatted_message = (
            f"{emoji} <b>{alert_level}: {title}</b>\n\n"
            f"{message}\n\n"
            f"<i>Time: {timestamp}</i>"
        )
        
        try:
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=formatted_message,
                parse_mode='HTML'
            )
            logger.info(f"Sent Telegram alert: [{alert_level}] {title}")
            return True
        except TelegramError as e:
            logger.error(f"Failed to send Telegram alert: {e}")
            return False
    
    def send_alert_sync(
        self,
        title: str,
        message: str,
        alert_level: str = 'INFO'
    ) -> bool:
        """
        Synchronous wrapper for send_alert
        
        Args:
            title: Alert title
            message: Alert message content
            alert_level: 'CRITICAL', 'WARNING', or 'INFO'
            
        Returns:
            True if sent successfully, False otherwise
        """
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(
            self.send_alert(title, message, alert_level)
        )
    
    async def send_var_breach_alert(
        self,
        var_level: float,
        threshold: float,
        confidence: float
    ) -> bool:
        """
        Send VaR breach alert
        
        Args:
            var_level: Current VaR level
            threshold: VaR threshold
            confidence: Confidence level (e.g., 0.95)
            
        Returns:
            True if sent successfully
        """
        title = f"VaR Breach Detected ({confidence*100:.0f}%)"
        message = (
            f"<b>Value at Risk (VaR) has exceeded the threshold!</b>\n\n"
            f"Current VaR: <code>{var_level:.2%}</code>\n"
            f"Threshold: <code>{threshold:.2%}</code>\n"
            f"Breach: <code>{((var_level - threshold) / threshold * 100):.1f}%</code>\n\n"
            f"<i>Consider reviewing portfolio positions and risk exposure.</i>"
        )
        return await self.send_alert(title, message, 'CRITICAL')
    
    async def send_drawdown_alert(
        self,
        current_drawdown: float,
        threshold: float
    ) -> bool:
        """
        Send drawdown alert
        
        Args:
            current_drawdown: Current drawdown level
            threshold: Drawdown threshold
            
        Returns:
            True if sent successfully
        """
        title = "Significant Drawdown Alert"
        message = (
            f"<b>Portfolio has experienced significant drawdown!</b>\n\n"
            f"Current Drawdown: <code>{current_drawdown:.2%}</code>\n"
            f"Threshold: <code>{threshold:.2%}</code>\n"
            f"Excess: <code>{((current_drawdown - threshold) / threshold * 100):.1f}%</code>\n\n"
            f"<i>Review stop-loss levels and consider risk mitigation strategies.</i>"
        )
        return await self.send_alert(title, message, 'CRITICAL')
    
    async def send_correlation_alert(
        self,
        high_corr_pairs: List[tuple],
        threshold: float
    ) -> bool:
        """
        Send correlation anomaly alert
        
        Args:
            high_corr_pairs: List of (asset1, asset2, correlation) tuples
            threshold: Correlation threshold
            
        Returns:
            True if sent successfully
        """
        title = "High Correlation Detected"
        
        pairs_text = "\n".join([
            f"• {asset1} - {asset2}: <code>{corr:.3f}</code>"
            for asset1, asset2, corr in high_corr_pairs[:5]
        ])
        
        message = (
            f"<b>High correlation detected between assets!</b>\n\n"
            f"Threshold: <code>{threshold:.2f}</code>\n\n"
            f"<b>Correlated Pairs:</b>\n{pairs_text}\n\n"
            f"<i>High correlation may reduce diversification benefits.</i>"
        )
        return await self.send_alert(title, message, 'WARNING')
    
    async def send_tail_risk_alert(
        self,
        skewness: float,
        kurtosis: float
    ) -> bool:
        """
        Send tail risk alert
        
        Args:
            skewness: Portfolio skewness
            kurtosis: Portfolio kurtosis
            
        Returns:
            True if sent successfully
        """
        title = "Tail Risk Warning"
        message = (
            f"<b>Elevated tail risk detected in portfolio!</b>\n\n"
            f"Skewness: <code>{skewness:.3f}</code>\n"
            f"Kurtosis: <code>{kurtosis:.3f}</code>\n\n"
            f"<i>Portfolio exhibits fat tails, indicating higher probability of extreme events.</i>"
        )
        return await self.send_alert(title, message, 'WARNING')
    
    async def send_stress_test_alert(
        self,
        scenario_name: str,
        portfolio_loss: float,
        var_breach: bool
    ) -> bool:
        """
        Send stress test alert
        
        Args:
            scenario_name: Name of stress scenario
            portfolio_loss: Portfolio loss percentage
            var_breach: Whether VaR was breached
            
        Returns:
            True if sent successfully
        """
        title = f"Stress Test: {scenario_name}"
        
        breach_text = "🚨 <b>VaR BREACH</b>" if var_breach else "✅ Within VaR"
        
        message = (
            f"<b>Stress test scenario results:</b>\n\n"
            f"Scenario: <code>{scenario_name}</code>\n"
            f"Portfolio Loss: <code>{portfolio_loss:.2%}</code>\n"
            f"Status: {breach_text}\n\n"
            f"<i>Review scenario impact and adjust positions if needed.</i>"
        )
        
        alert_level = 'CRITICAL' if var_breach else 'INFO'
        return await self.send_alert(title, message, alert_level)
    
    async def send_daily_summary(
        self,
        metrics: Dict[str, float],
        alerts: Dict[str, List[str]]
    ) -> bool:
        """
        Send daily risk summary
        
        Args:
            metrics: Dictionary of risk metrics
            alerts: Dictionary of alerts by level
            
        Returns:
            True if sent successfully
        """
        title = "Daily Risk Summary"
        
        # Format metrics
        metrics_text = "\n".join([
            f"• {key}: <code>{value:.2%}</code>" if isinstance(value, float) and abs(value) < 1
            else f"• {key}: <code>{value:.2f}</code>"
            for key, value in metrics.items()
        ])
        
        # Format alerts
        alert_counts = {
            level: len(msgs) for level, msgs in alerts.items()
            if msgs
        }
        
        if alert_counts:
            alerts_text = "\n".join([
                f"🚨 Critical: {alert_counts.get('critical', 0)}",
                f"⚠️ Warnings: {alert_counts.get('warning', 0)}",
                f"ℹ️ Info: {alert_counts.get('info', 0)}"
            ])
        else:
            alerts_text = "✅ No alerts"
        
        message = (
            f"<b>Portfolio Risk Summary</b>\n\n"
            f"<b>Key Metrics:</b>\n{metrics_text}\n\n"
            f"<b>Alerts:</b>\n{alerts_text}\n\n"
            f"<i>Daily summary generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</i>"
        )
        
        return await self.send_alert(title, message, 'INFO')
    
    async def test_connection(self) -> bool:
        """
        Test Telegram connection
        
        Returns:
            True if connection successful
        """
        return await self.send_alert(
            "Connection Test",
            "Telegram alert system is configured and working correctly.",
            'INFO'
        )


def example_usage():
    """Example usage of TelegramAlerter"""
    # Initialize alerter
    alerter = TelegramAlerter()
    
    if not alerter.enabled:
        print("Telegram not configured. Set environment variables:")
        print("  export TELEGRAM_BOT_TOKEN='your_bot_token'")
        print("  export TELEGRAM_CHAT_ID='your_chat_id'")
        return
    
    # Test connection
    asyncio.run(alerter.test_connection())
    
    # Example alerts
    asyncio.run(alerter.send_var_breach_alert(
        var_level=0.08,
        threshold=0.05,
        confidence=0.95
    ))
    
    asyncio.run(alerter.send_drawdown_alert(
        current_drawdown=-0.25,
        threshold=-0.20
    ))


if __name__ == "__main__":
    example_usage()
