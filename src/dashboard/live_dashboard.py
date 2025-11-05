"""
Live Paper Trading Dashboard - Monitor bot in real-time
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path
from datetime import datetime
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.strategies import AVAILABLE_STRATEGIES
from src.paper_trading import LiveTradingBot

# Page config
st.set_page_config(
    page_title="Live Paper Trading Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'bot' not in st.session_state:
    st.session_state.bot = None
if 'bot_thread' not in st.session_state:
    st.session_state.bot_thread = None

# Title
st.title("🤖 Live Paper Trading Dashboard")
st.markdown("---")

# Sidebar - Bot Configuration
st.sidebar.header("⚙️ Bot Configuration")

# Strategy Selection
st.sidebar.subheader("🎯 Strategy")
selected_strategy_name = st.sidebar.selectbox(
    "Select Strategy",
    list(AVAILABLE_STRATEGIES.keys()),
    index=0
)

# Symbol and Timeframe
st.sidebar.subheader("📊 Market")
symbol = st.sidebar.text_input("Trading Pair", value="BTC/USDT")
timeframe = st.sidebar.selectbox("Timeframe", ['1m', '5m', '15m', '1h', '4h'], index=3)
exchange = st.sidebar.selectbox("Exchange", ['binance', 'kraken', 'coinbase'], index=0)

# Bot Settings
st.sidebar.subheader("💰 Capital & Risk")
initial_capital = st.sidebar.number_input("Initial Capital ($)", min_value=100, value=10000, step=100)
fees = st.sidebar.number_input("Fees (%)", min_value=0.0, max_value=1.0, value=0.1, step=0.01)
slippage = st.sidebar.number_input("Slippage (%)", min_value=0.0, max_value=1.0, value=0.05, step=0.01)
check_interval = st.sidebar.slider("Check Interval (seconds)", 10, 300, 60)

# Strategy Parameters
st.sidebar.subheader("🔧 Strategy Parameters")
strategy_class = AVAILABLE_STRATEGIES[selected_strategy_name]
temp_instance = strategy_class()
param_schema = temp_instance.get_param_schema()

strategy_params = {}
for param_name, param_config in param_schema.items():
    if param_config['type'] == 'int':
        value = st.sidebar.slider(
            param_config['label'],
            min_value=param_config['min'],
            max_value=param_config['max'],
            value=param_config['default'],
            key=f"live_{param_name}"
        )
        strategy_params[param_name] = value

# Bot Control Buttons
st.sidebar.markdown("---")
st.sidebar.subheader("🎮 Bot Control")

col1, col2 = st.sidebar.columns(2)

if col1.button("▶️ Start Bot", type="primary", disabled=st.session_state.bot is not None):
    # Create strategy instance
    strategy = strategy_class(**strategy_params)

    # Create bot
    bot = LiveTradingBot(
        strategy=strategy,
        symbol=symbol,
        timeframe=timeframe,
        exchange_id=exchange,
        initial_capital=initial_capital,
        fees_pct=fees,
        slippage_pct=slippage,
        check_interval=check_interval
    )

    # Store in session state
    st.session_state.bot = bot

    # Run in background
    st.session_state.bot_thread = bot.run_async()

    st.sidebar.success(f"✅ Bot started!")
    time.sleep(1)
    st.rerun()

if col2.button("⏹️ Stop Bot", type="secondary", disabled=st.session_state.bot is None):
    if st.session_state.bot:
        st.session_state.bot.stop()
        st.session_state.bot = None
        st.session_state.bot_thread = None
        st.sidebar.warning("⏹️ Bot stopped")
        time.sleep(1)
        st.rerun()

# Main Dashboard
if st.session_state.bot is None:
    # No bot running - show welcome screen
    st.info("👈 Configure your bot settings and click '▶️ Start Bot' to begin paper trading")

    st.markdown("---")
    st.subheader("📚 Available Strategies")

    for name, strategy_class in AVAILABLE_STRATEGIES.items():
        temp = strategy_class()
        with st.expander(f"🎯 {name}"):
            st.markdown(f"**Description:** {temp.get_description()}")
            st.markdown("**Parameters:**")
            for param, config in temp.get_param_schema().items():
                st.markdown(f"- {config['label']}: {config['default']} (range: {config['min']}-{config['max']})")

else:
    # Bot is running - show live dashboard
    bot = st.session_state.bot

    # Auto-refresh
    placeholder = st.empty()

    with placeholder.container():
        # Get bot status
        status = bot.get_status()

        # Status Bar
        col1, col2, col3, col4, col5 = st.columns(5)

        col1.metric(
            "🤖 Status",
            "RUNNING" if status['is_running'] else "STOPPED",
            delta="Live" if status['is_running'] else None
        )

        col2.metric(
            "💰 Portfolio Value",
            f"${status['total_value']:,.2f}",
            delta=f"{status['total_pnl_pct']:+.2f}%"
        )

        col3.metric(
            "📊 P&L",
            f"${status['total_pnl']:,.2f}",
            delta=f"{status['total_pnl_pct']:+.2f}%"
        )

        col4.metric(
            "🔄 Trades",
            status['num_trades'],
            delta=f"{status['win_rate']:.1f}% WR"
        )

        col5.metric(
            "📍 Positions",
            status['num_open_positions']
        )

        st.markdown("---")

        # Main Content
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Chart", "💼 Portfolio", "📋 Trades", "📊 Stats"])

        with tab1:
            st.subheader(f"📈 {bot.symbol} - Live Chart")

            if not bot.historical_data.empty:
                df = bot.historical_data

                # Candlestick chart
                fig = make_subplots(
                    rows=2, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.03,
                    row_heights=[0.7, 0.3]
                )

                fig.add_trace(
                    go.Candlestick(
                        x=df.index,
                        open=df['open'],
                        high=df['high'],
                        low=df['low'],
                        close=df['close'],
                        name='Price'
                    ),
                    row=1, col=1
                )

                # Volume
                fig.add_trace(
                    go.Bar(x=df.index, y=df['volume'], name='Volume', marker_color='rgba(128,128,128,0.3)'),
                    row=2, col=1
                )

                fig.update_layout(
                    height=600,
                    showlegend=False,
                    xaxis_rangeslider_visible=False
                )

                fig.update_xaxes(title_text="Time", row=2, col=1)
                fig.update_yaxes(title_text="Price ($)", row=1, col=1)
                fig.update_yaxes(title_text="Volume", row=2, col=1)

                st.plotly_chart(fig, use_container_width=True)

                # Current Price Info
                current_price = df['close'].iloc[-1]
                price_change = ((df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0]) * 100

                col1, col2, col3 = st.columns(3)
                col1.metric("Current Price", f"${current_price:,.2f}")
                col2.metric("24h Change", f"{price_change:+.2f}%")
                col3.metric("Last Update", status['last_check'].strftime("%H:%M:%S") if status['last_check'] else "N/A")

            else:
                st.info("Waiting for data...")

        with tab2:
            st.subheader("💼 Open Positions")

            if bot.engine.portfolio.positions:
                positions_data = []
                for pos in bot.engine.portfolio.positions:
                    positions_data.append({
                        'Symbol': pos.symbol,
                        'Side': pos.side.value.upper(),
                        'Quantity': f"{pos.quantity:.6f}",
                        'Entry Price': f"${pos.entry_price:,.2f}",
                        'Current Price': f"${pos.current_price:,.2f}",
                        'Unrealized P&L': f"${pos.unrealized_pnl:,.2f}",
                        'Unrealized P&L %': f"{pos.unrealized_pnl_pct:+.2f}%",
                        'Entry Time': pos.entry_timestamp.strftime("%Y-%m-%d %H:%M:%S")
                    })

                st.dataframe(pd.DataFrame(positions_data), use_container_width=True, hide_index=True)
            else:
                st.info("No open positions")

            # Capital Info
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            col1.metric("💵 Available Capital", f"${status['current_capital']:,.2f}")
            col2.metric("💰 Total Value", f"${status['total_value']:,.2f}")
            col3.metric("💸 Total Fees", f"${status['total_fees']:,.2f}")

        with tab3:
            st.subheader("📋 Trade History")

            if bot.engine.portfolio.trades:
                trades_data = []
                for trade in bot.engine.portfolio.trades:
                    trades_data.append({
                        'ID': trade.id,
                        'Symbol': trade.symbol,
                        'Side': trade.side.value.upper(),
                        'Entry': f"${trade.entry_price:,.2f}",
                        'Exit': f"${trade.exit_price:,.2f}",
                        'Quantity': f"{trade.quantity:.6f}",
                        'P&L': f"${trade.pnl:,.2f}",
                        'P&L %': f"{trade.pnl_pct:+.2f}%",
                        'Duration': f"{trade.duration_seconds/3600:.1f}h",
                        'Entry Time': trade.entry_timestamp.strftime("%Y-%m-%d %H:%M"),
                        'Exit Time': trade.exit_timestamp.strftime("%Y-%m-%d %H:%M")
                    })

                st.dataframe(
                    pd.DataFrame(trades_data).sort_values('Exit Time', ascending=False),
                    use_container_width=True,
                    hide_index=True
                )

                # Trade statistics
                st.markdown("---")
                col1, col2, col3, col4 = st.columns(4)

                winning_trades = [t for t in bot.engine.portfolio.trades if t.is_winning]
                losing_trades = [t for t in bot.engine.portfolio.trades if not t.is_winning]

                col1.metric("✅ Winning Trades", len(winning_trades))
                col2.metric("❌ Losing Trades", len(losing_trades))

                if winning_trades:
                    avg_win = sum(t.pnl for t in winning_trades) / len(winning_trades)
                    col3.metric("💚 Avg Win", f"${avg_win:,.2f}")

                if losing_trades:
                    avg_loss = sum(t.pnl for t in losing_trades) / len(losing_trades)
                    col4.metric("💔 Avg Loss", f"${avg_loss:,.2f}")

            else:
                st.info("No completed trades yet")

        with tab4:
            st.subheader("📊 Performance Statistics")

            # Performance metrics
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### 💰 Returns")
                st.metric("Initial Capital", f"${bot.engine.initial_capital:,.2f}")
                st.metric("Current Value", f"${status['total_value']:,.2f}")
                st.metric("Total P&L", f"${status['total_pnl']:,.2f}", delta=f"{status['total_pnl_pct']:+.2f}%")
                st.metric("Total Fees Paid", f"${status['total_fees']:,.2f}")

            with col2:
                st.markdown("### 📈 Trading Activity")
                st.metric("Total Trades", status['num_trades'])
                st.metric("Win Rate", f"{status['win_rate']:.2f}%")
                st.metric("Open Positions", status['num_open_positions'])
                st.metric("Total Signal Checks", status['total_checks'])
                st.metric("Signals Generated", status['total_signals'])

            # Bot Info
            st.markdown("---")
            st.markdown("### 🤖 Bot Information")

            info_col1, info_col2 = st.columns(2)

            with info_col1:
                st.markdown(f"**Strategy:** {status['strategy']}")
                st.markdown(f"**Symbol:** {status['symbol']}")
                st.markdown(f"**Timeframe:** {bot.timeframe}")
                st.markdown(f"**Exchange:** {bot.exchange.id}")

            with info_col2:
                st.markdown(f"**Check Interval:** {bot.check_interval}s")
                st.markdown(f"**Fees:** {fees}%")
                st.markdown(f"**Slippage:** {slippage}%")
                if status['last_check']:
                    st.markdown(f"**Last Check:** {status['last_check'].strftime('%Y-%m-%d %H:%M:%S')}")

    # Auto-refresh
    time.sleep(5)
    st.rerun()
