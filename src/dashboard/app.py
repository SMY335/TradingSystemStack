"""
Trading Bot Dashboard - Streamlit App
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.strategies import AVAILABLE_STRATEGIES, EMAStrategy, RSIStrategy, MACDStrategy
from src.data_sources import CryptoDataFetcher
from src.backtesting import BacktestEngine


# Page config
st.set_page_config(
    page_title="Trading Bot Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title
st.title("📈 Trading Bot Dashboard")
st.markdown("---")


# Sidebar - Configuration
st.sidebar.header("⚙️ Configuration")

# Data Source Selection
st.sidebar.subheader("📊 Data Source")
exchange = st.sidebar.selectbox(
    "Exchange",
    CryptoDataFetcher.get_supported_exchanges(),
    index=0
)

symbol = st.sidebar.text_input("Trading Pair", value="BTC/USDT")
timeframe = st.sidebar.selectbox("Timeframe", ['1h', '4h', '1d', '15m', '5m'], index=0)
days_back = st.sidebar.slider("Days of History", 7, 365, 90)

# Backtest Configuration
st.sidebar.subheader("💰 Backtest Settings")
initial_cash = st.sidebar.number_input("Initial Capital ($)", min_value=100, value=10000, step=100)
fees = st.sidebar.number_input("Fees (%)", min_value=0.0, max_value=1.0, value=0.1, step=0.01) / 100
slippage = st.sidebar.number_input("Slippage (%)", min_value=0.0, max_value=1.0, value=0.05, step=0.01) / 100

# Strategy Selection
st.sidebar.subheader("🎯 Strategy Selection")
selected_strategies = st.sidebar.multiselect(
    "Choose Strategies",
    list(AVAILABLE_STRATEGIES.keys()),
    default=['EMA Crossover']
)

# Strategy Parameters (Dynamic based on selection)
strategy_params = {}
for strategy_name in selected_strategies:
    st.sidebar.markdown(f"**{strategy_name} Parameters:**")
    strategy_class = AVAILABLE_STRATEGIES[strategy_name]
    temp_instance = strategy_class()
    param_schema = temp_instance.get_param_schema()

    strategy_params[strategy_name] = {}
    for param_name, param_config in param_schema.items():
        if param_config['type'] == 'int':
            value = st.sidebar.slider(
                param_config['label'],
                min_value=param_config['min'],
                max_value=param_config['max'],
                value=param_config['default'],
                key=f"{strategy_name}_{param_name}"
            )
            strategy_params[strategy_name][param_name] = value

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader(f"📈 {symbol} Price Chart")

with col2:
    st.subheader("📊 Quick Stats")


# Fetch Data Button
if st.sidebar.button("🚀 Run Backtest", type="primary"):
    with st.spinner(f"Fetching {symbol} data from {exchange}..."):
        try:
            # Fetch data
            fetcher = CryptoDataFetcher(exchange)
            df = fetcher.fetch_ohlcv(symbol, timeframe, days_back)

            st.success(f"✓ Loaded {len(df)} candles")

            # Display price chart
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                row_heights=[0.7, 0.3]
            )

            # Candlestick chart
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

            fig.update_xaxes(title_text="Date", row=2, col=1)
            fig.update_yaxes(title_text="Price ($)", row=1, col=1)
            fig.update_yaxes(title_text="Volume", row=2, col=1)

            col1.plotly_chart(fig, use_container_width=True)

            # Display current stats
            latest_price = df['close'].iloc[-1]
            price_change = ((df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0]) * 100

            col2.metric("Current Price", f"${latest_price:,.2f}")
            col2.metric("Period Change", f"{price_change:,.2f}%", delta=f"{price_change:,.2f}%")
            col2.metric("Highest", f"${df['high'].max():,.2f}")
            col2.metric("Lowest", f"${df['low'].min():,.2f}")

            # Run backtests
            if selected_strategies:
                st.markdown("---")
                st.subheader("🎯 Backtest Results")

                # Initialize backtest engine
                engine = BacktestEngine(
                    initial_cash=initial_cash,
                    fees=fees,
                    slippage=slippage
                )

                # Create strategy instances with custom parameters
                strategies = []
                for strategy_name in selected_strategies:
                    strategy_class = AVAILABLE_STRATEGIES[strategy_name]
                    params = strategy_params.get(strategy_name, {})
                    strategies.append(strategy_class(**params))

                # Compare strategies
                with st.spinner("Running backtests..."):
                    results_df = engine.compare_strategies(strategies, df)

                    # Display results table
                    st.dataframe(
                        results_df,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "total_return_pct": st.column_config.NumberColumn("Return (%)", format="%.2f"),
                            "win_rate_pct": st.column_config.NumberColumn("Win Rate (%)", format="%.2f"),
                            "profit_factor": st.column_config.NumberColumn("Profit Factor", format="%.2f"),
                            "max_drawdown_pct": st.column_config.NumberColumn("Max DD (%)", format="%.2f"),
                            "sharpe_ratio": st.column_config.NumberColumn("Sharpe", format="%.2f"),
                            "total_trades": st.column_config.NumberColumn("Trades", format="%d"),
                            "final_value": st.column_config.NumberColumn("Final Value ($)", format="$%.2f"),
                            "total_pnl": st.column_config.NumberColumn("P&L ($)", format="$%.2f"),
                        }
                    )

                    # Show detailed results for each strategy
                    st.markdown("---")
                    st.subheader("📊 Detailed Performance")

                    for strategy in strategies:
                        with st.expander(f"📈 {strategy.name} - {strategy.get_description()}"):
                            portfolio, kpis = engine.run(strategy, df)

                            # Metrics in columns
                            metric_cols = st.columns(4)
                            metric_cols[0].metric("Total Return", f"{kpis['total_return_pct']}%")
                            metric_cols[1].metric("Win Rate", f"{kpis['win_rate_pct']}%")
                            metric_cols[2].metric("Total Trades", kpis['total_trades'])
                            metric_cols[3].metric("Profit Factor", kpis['profit_factor'])

                            # Equity curve
                            equity = portfolio.value()
                            equity_fig = go.Figure()
                            equity_fig.add_trace(go.Scatter(
                                x=equity.index,
                                y=equity.values,
                                mode='lines',
                                name='Portfolio Value',
                                line=dict(color='green', width=2)
                            ))

                            equity_fig.update_layout(
                                title="Equity Curve",
                                xaxis_title="Date",
                                yaxis_title="Portfolio Value ($)",
                                height=400
                            )

                            st.plotly_chart(equity_fig, use_container_width=True)
            else:
                st.warning("Please select at least one strategy to backtest")

        except Exception as e:
            st.error(f"Error: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
else:
    st.info("👈 Configure your settings and click 'Run Backtest' to get started")

    # Show sample strategies
    st.markdown("---")
    st.subheader("📚 Available Strategies")

    for name, strategy_class in AVAILABLE_STRATEGIES.items():
        temp_instance = strategy_class()
        st.markdown(f"**{name}**: {temp_instance.get_description()}")
