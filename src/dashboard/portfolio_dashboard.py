"""
Portfolio Management Dashboard
Interactive Streamlit dashboard for portfolio optimization and rebalancing
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sys
sys.path.insert(0, '../')

from portfolio.models import Portfolio, Asset, Position, AssetType
from portfolio.optimizer import PortfolioOptimizer, OptimizationMethod, RiskMeasure
from portfolio.portfolio_manager import PortfolioManager, RebalancingConfig
from data_sources.crypto_data import CryptoDataSource


# Page config
st.set_page_config(
    page_title="Portfolio Manager",
    page_icon="💼",
    layout="wide"
)

st.title("💼 Portfolio Management Dashboard")
st.markdown("---")


@st.cache_resource
def get_data_source():
    """Initialize data source"""
    return CryptoDataSource()


def create_sample_portfolio():
    """Create a sample portfolio for demonstration"""
    portfolio = Portfolio(
        name="Sample Multi-Asset Portfolio",
        initial_capital=100000,
        cash=10000
    )

    # Sample positions
    positions_data = [
        ("BTC/USDT", 1.5, 45000, AssetType.CRYPTO),
        ("ETH/USDT", 25, 2500, AssetType.CRYPTO),
        ("BNB/USDT", 150, 350, AssetType.CRYPTO),
        ("SOL/USDT", 500, 85, AssetType.CRYPTO),
    ]

    for symbol, qty, price, asset_type in positions_data:
        asset = Asset(symbol, asset_type, "binance")
        pos = Position(
            asset=asset,
            quantity=qty,
            entry_price=price,
            entry_date=datetime.now() - timedelta(days=30),
            current_price=price
        )
        portfolio.add_position(pos)

    return portfolio


def fetch_data_function(symbols, days):
    """Fetch historical data for optimization"""
    data_source = get_data_source()

    # Fetch data for each symbol
    dfs = []
    for symbol in symbols:
        try:
            df = data_source.fetch_ohlcv(symbol, '1d', days=days)
            if not df.empty:
                df = df[['close']].rename(columns={'close': symbol})
                dfs.append(df)
        except Exception as e:
            st.warning(f"Could not fetch data for {symbol}: {e}")

    if not dfs:
        return pd.DataFrame()

    # Merge all dataframes
    result = dfs[0]
    for df in dfs[1:]:
        result = result.join(df, how='outer')

    # Calculate returns
    returns = result.pct_change().dropna()
    return returns


# Sidebar - Portfolio Selection
with st.sidebar:
    st.header("⚙️ Settings")

    portfolio_option = st.selectbox(
        "Portfolio",
        ["Sample Portfolio", "Create New"]
    )

    if portfolio_option == "Sample Portfolio":
        portfolio = create_sample_portfolio()
    else:
        st.info("Custom portfolio creation coming soon!")
        portfolio = create_sample_portfolio()

    st.markdown("---")
    st.subheader("📊 Portfolio Overview")
    st.metric("Total Value", f"${portfolio.total_value:,.2f}")
    st.metric("Total Return", f"{portfolio.total_return_pct:.2f}%")
    st.metric("Cash", f"${portfolio.cash:,.2f}")
    st.metric("Positions", len(portfolio.positions))


# Main content tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Overview",
    "🎯 Optimization",
    "⚖️ Rebalancing",
    "📈 Analytics"
])

# TAB 1: Portfolio Overview
with tab1:
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Portfolio Composition")

        # Create pie chart of weights
        weights = portfolio.weights
        if weights:
            fig = px.pie(
                values=list(weights.values()),
                names=list(weights.keys()),
                title="Portfolio Allocation",
                hole=0.4
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Key Metrics")

        metrics_data = {
            "Metric": ["Initial Capital", "Current Value", "Cash", "Invested", "Total P&L", "Total Return %"],
            "Value": [
                f"${portfolio.initial_capital:,.2f}",
                f"${portfolio.total_value:,.2f}",
                f"${portfolio.cash:,.2f}",
                f"${portfolio.total_market_value:,.2f}",
                f"${portfolio.total_pnl:,.2f}",
                f"{portfolio.total_return_pct:.2f}%"
            ]
        }
        st.table(pd.DataFrame(metrics_data))

    st.subheader("Positions Detail")

    positions_data = []
    for pos in portfolio.positions:
        positions_data.append({
            "Symbol": pos.asset.symbol,
            "Quantity": f"{pos.quantity:.4f}",
            "Entry Price": f"${pos.entry_price:,.2f}",
            "Current Price": f"${pos.current_price or pos.entry_price:,.2f}",
            "Market Value": f"${pos.market_value:,.2f}",
            "P&L": f"${pos.unrealized_pnl:,.2f}",
            "P&L %": f"{pos.unrealized_pnl_pct:+.2f}%",
            "Weight": f"{weights.get(pos.asset.symbol, 0)*100:.2f}%"
        })

    if positions_data:
        st.dataframe(pd.DataFrame(positions_data), use_container_width=True)
    else:
        st.info("No positions in portfolio")


# TAB 2: Portfolio Optimization
with tab2:
    st.subheader("🎯 Portfolio Optimization")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("#### Optimization Settings")

        opt_method = st.selectbox(
            "Optimization Method",
            [
                "Maximum Sharpe Ratio",
                "Minimum Volatility",
                "Risk Parity",
                "Equal Weight"
            ]
        )

        risk_measure_option = st.selectbox(
            "Risk Measure",
            ["Standard Deviation", "CVaR", "Mean Absolute Deviation"]
        )

        lookback_days = st.slider(
            "Lookback Period (days)",
            min_value=30,
            max_value=365,
            value=90,
            step=30
        )

        optimize_button = st.button("🚀 Optimize Portfolio", type="primary")

    with col2:
        if optimize_button:
            with st.spinner("Optimizing portfolio..."):
                try:
                    # Map selections to enums
                    method_map = {
                        "Maximum Sharpe Ratio": OptimizationMethod.MAX_SHARPE,
                        "Minimum Volatility": OptimizationMethod.MIN_VOLATILITY,
                        "Risk Parity": OptimizationMethod.RISK_PARITY,
                        "Equal Weight": OptimizationMethod.EQUAL_WEIGHT
                    }

                    risk_map = {
                        "Standard Deviation": RiskMeasure.MV,
                        "CVaR": RiskMeasure.CVaR,
                        "Mean Absolute Deviation": RiskMeasure.MAD
                    }

                    # Get symbols
                    symbols = [pos.asset.symbol for pos in portfolio.positions]

                    if len(symbols) < 2:
                        st.error("Need at least 2 assets for optimization")
                    else:
                        # Fetch data
                        returns_df = fetch_data_function(symbols, lookback_days)

                        if returns_df.empty:
                            st.error("Failed to fetch historical data")
                        else:
                            # Optimize
                            optimizer = PortfolioOptimizer()
                            optimal_weights = optimizer.optimize(
                                returns_df,
                                method=method_map[opt_method],
                                risk_measure=risk_map[risk_measure_option]
                            )

                            # Calculate metrics
                            current_weights = {k: v for k, v in portfolio.weights.items() if k != 'CASH'}
                            current_metrics = optimizer.calculate_portfolio_metrics(
                                current_weights, returns_df
                            )
                            optimal_metrics = optimizer.calculate_portfolio_metrics(
                                optimal_weights, returns_df
                            )

                            # Display results
                            st.success("✅ Optimization Complete!")

                            # Metrics comparison
                            st.markdown("#### Performance Comparison")

                            metrics_comparison = pd.DataFrame({
                                "Metric": ["Expected Return", "Volatility", "Sharpe Ratio", "Max Drawdown"],
                                "Current": [
                                    f"{current_metrics['expected_return']*100:.2f}%",
                                    f"{current_metrics['volatility']*100:.2f}%",
                                    f"{current_metrics['sharpe_ratio']:.3f}",
                                    f"{current_metrics['max_drawdown']*100:.2f}%"
                                ],
                                "Optimal": [
                                    f"{optimal_metrics['expected_return']*100:.2f}%",
                                    f"{optimal_metrics['volatility']*100:.2f}%",
                                    f"{optimal_metrics['sharpe_ratio']:.3f}",
                                    f"{optimal_metrics['max_drawdown']*100:.2f}%"
                                ],
                                "Improvement": [
                                    f"{(optimal_metrics['expected_return'] - current_metrics['expected_return'])*100:+.2f}%",
                                    f"{(optimal_metrics['volatility'] - current_metrics['volatility'])*100:+.2f}%",
                                    f"{(optimal_metrics['sharpe_ratio'] - current_metrics['sharpe_ratio']):+.3f}",
                                    f"{(optimal_metrics['max_drawdown'] - current_metrics['max_drawdown'])*100:+.2f}%"
                                ]
                            })

                            st.dataframe(metrics_comparison, use_container_width=True)

                            # Weights comparison
                            st.markdown("#### Optimal Weights")

                            col_a, col_b = st.columns(2)

                            with col_a:
                                # Current weights pie
                                fig1 = px.pie(
                                    values=list(current_weights.values()),
                                    names=list(current_weights.keys()),
                                    title="Current Weights"
                                )
                                st.plotly_chart(fig1, use_container_width=True)

                            with col_b:
                                # Optimal weights pie
                                fig2 = px.pie(
                                    values=list(optimal_weights.values()),
                                    names=list(optimal_weights.keys()),
                                    title="Optimal Weights"
                                )
                                st.plotly_chart(fig2, use_container_width=True)

                            # Store optimal weights in session state
                            st.session_state['optimal_weights'] = optimal_weights
                            st.session_state['optimal_metrics'] = optimal_metrics

                except Exception as e:
                    st.error(f"Optimization error: {e}")
                    import traceback
                    st.code(traceback.format_exc())
        else:
            st.info("Configure optimization settings and click 'Optimize Portfolio' to see results")


# TAB 3: Rebalancing
with tab3:
    st.subheader("⚖️ Portfolio Rebalancing")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("#### Rebalancing Settings")

        rebal_frequency = st.selectbox(
            "Frequency",
            ["daily", "weekly", "monthly", "quarterly"]
        )

        drift_threshold = st.slider(
            "Drift Threshold (%)",
            min_value=1,
            max_value=20,
            value=5,
            step=1
        ) / 100

        rebalance_button = st.button("⚖️ Calculate Rebalancing", type="primary")

    with col2:
        if rebalance_button:
            with st.spinner("Calculating rebalancing..."):
                try:
                    # Create portfolio manager
                    config = RebalancingConfig(
                        frequency=rebal_frequency,
                        threshold=drift_threshold,
                        optimization_method=OptimizationMethod.MAX_SHARPE,
                        lookback_period=90
                    )

                    manager = PortfolioManager(
                        portfolio=portfolio,
                        data_fetcher=fetch_data_function,
                        rebalancing_config=config
                    )

                    # Execute rebalancing
                    result = manager.rebalance(force=True)

                    if result['rebalanced']:
                        st.success("✅ Rebalancing Calculated!")

                        # Display trades
                        st.markdown("#### Required Trades")

                        trades_data = []
                        for symbol, weight_change in result['trades'].items():
                            dollar_change = weight_change * portfolio.total_value
                            trades_data.append({
                                "Symbol": symbol,
                                "Weight Change": f"{weight_change*100:+.2f}%",
                                "Dollar Amount": f"${dollar_change:+,.2f}",
                                "Action": "BUY" if weight_change > 0 else "SELL"
                            })

                        if trades_data:
                            st.dataframe(pd.DataFrame(trades_data), use_container_width=True)
                        else:
                            st.info("No trades needed - portfolio is already optimal!")

                        # Performance improvement
                        st.markdown("#### Expected Improvement")
                        col_a, col_b = st.columns(2)

                        with col_a:
                            st.metric(
                                "Sharpe Ratio Change",
                                f"{result['improvement']['sharpe_ratio']:+.3f}"
                            )

                        with col_b:
                            st.metric(
                                "Volatility Change",
                                f"{result['improvement']['volatility']*100:+.2f}%"
                            )

                    else:
                        st.info(f"No rebalancing needed: {result['reason']}")

                except Exception as e:
                    st.error(f"Rebalancing error: {e}")
                    import traceback
                    st.code(traceback.format_exc())
        else:
            st.info("Configure settings and click 'Calculate Rebalancing' to see required trades")


# TAB 4: Analytics
with tab4:
    st.subheader("📈 Portfolio Analytics")

    analysis_period = st.slider(
        "Analysis Period (days)",
        min_value=30,
        max_value=365,
        value=90,
        step=30
    )

    if st.button("📊 Analyze Performance"):
        with st.spinner("Analyzing portfolio..."):
            try:
                # Create manager
                manager = PortfolioManager(
                    portfolio=portfolio,
                    data_fetcher=fetch_data_function
                )

                # Get analysis
                analysis = manager.analyze_performance(lookback_days=analysis_period)

                if 'error' in analysis:
                    st.error(analysis['error'])
                else:
                    st.success("✅ Analysis Complete!")

                    # Display metrics
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.metric("Expected Return", f"{analysis['expected_return']*100:.2f}%")
                        st.metric("Volatility", f"{analysis['volatility']*100:.2f}%")

                    with col2:
                        st.metric("Sharpe Ratio", f"{analysis['sharpe_ratio']:.3f}")
                        st.metric("Sortino Ratio", f"{analysis['sortino_ratio']:.3f}")

                    with col3:
                        st.metric("Max Drawdown", f"{analysis['max_drawdown']*100:.2f}%")
                        st.metric("Calmar Ratio", f"{analysis['calmar_ratio']:.3f}")

                    with col4:
                        st.metric("VaR (95%)", f"{analysis['var_95']*100:.2f}%")
                        st.metric("CVaR (95%)", f"{analysis['cvar_95']*100:.2f}%")

                    # Additional info
                    st.markdown("---")
                    st.markdown("#### Portfolio Info")
                    info_col1, info_col2 = st.columns(2)

                    with info_col1:
                        st.write(f"**Total Value:** ${analysis['total_value']:,.2f}")
                        st.write(f"**Total Return:** {analysis['total_return_pct']:.2f}%")

                    with info_col2:
                        st.write(f"**Cash Ratio:** {analysis['cash_ratio']*100:.2f}%")
                        st.write(f"**Analysis Period:** {analysis_period} days")

            except Exception as e:
                st.error(f"Analysis error: {e}")
                import traceback
                st.code(traceback.format_exc())


# Footer
st.markdown("---")
st.markdown(
    "**Portfolio Management Dashboard** | Powered by Riskfolio-Lib | "
    f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
)
