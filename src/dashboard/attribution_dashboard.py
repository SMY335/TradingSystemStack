"""
Performance Attribution Dashboard
Interactive Streamlit dashboard for performance attribution analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.portfolio.performance_attribution import PerformanceAttributor, generate_attribution_report
from src.data_sources.crypto_data import CryptoDataSource


def plot_brinson_attribution(brinson):
    """Plot Brinson attribution effects"""
    effects = {
        'Allocation': brinson.allocation_effect * 100,
        'Selection': brinson.selection_effect * 100,
        'Interaction': brinson.interaction_effect * 100
    }

    fig = go.Figure()

    # Waterfall chart
    fig.add_trace(go.Waterfall(
        name="Attribution",
        orientation="v",
        measure=["relative", "relative", "relative", "total"],
        x=["Allocation", "Selection", "Interaction", "Active Return"],
        y=[
            effects['Allocation'],
            effects['Selection'],
            effects['Interaction'],
            brinson.active_return * 100
        ],
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        decreasing={"marker": {"color": "red"}},
        increasing={"marker": {"color": "green"}},
        totals={"marker": {"color": "blue"}}
    ))

    fig.update_layout(
        title="Brinson Attribution - Active Return Decomposition",
        yaxis_title="Contribution (%)",
        height=400
    )

    return fig


def plot_asset_contribution(asset_contributions):
    """Plot asset contributions to return"""
    assets = list(asset_contributions.keys())
    allocations = [v['allocation'] * 100 for v in asset_contributions.values()]
    selections = [v['selection'] * 100 for v in asset_contributions.values()]
    totals = [v['total'] * 100 for v in asset_contributions.values()]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name='Allocation Effect',
        x=assets,
        y=allocations,
        marker_color='lightblue'
    ))

    fig.add_trace(go.Bar(
        name='Selection Effect',
        x=assets,
        y=selections,
        marker_color='lightgreen'
    ))

    fig.add_trace(go.Bar(
        name='Total',
        x=assets,
        y=totals,
        marker_color='orange',
        visible='legendonly'
    ))

    fig.update_layout(
        title="Asset Contribution to Active Return",
        xaxis_title="Asset",
        yaxis_title="Contribution (%)",
        barmode='group',
        height=400
    )

    return fig


def plot_risk_contribution(risk_attr):
    """Plot risk contribution by asset"""
    assets = list(risk_attr.risk_contribution_pct.keys())
    contributions = [v * 100 for v in risk_attr.risk_contribution_pct.values()]

    fig = go.Figure()

    fig.add_trace(go.Pie(
        labels=assets,
        values=contributions,
        hole=0.3,
        marker=dict(colors=px.colors.qualitative.Set3)
    ))

    fig.update_layout(
        title=f"Risk Contribution by Asset (Portfolio Risk: {risk_attr.portfolio_risk:.2%})",
        height=400
    )

    return fig


def plot_performance_comparison(metrics, benchmark_return):
    """Plot performance comparison"""
    categories = ['Total Return', 'Annualized Return', 'Sharpe Ratio', 'Max Drawdown']
    portfolio_values = [
        metrics.total_return * 100,
        metrics.annualized_return * 100,
        metrics.sharpe_ratio,
        abs(metrics.max_drawdown) * 100
    ]

    benchmark_values = [
        benchmark_return * 100,
        benchmark_return * 100,  # Simplified
        0,  # Placeholder
        0   # Placeholder
    ]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name='Portfolio',
        x=categories,
        y=portfolio_values,
        marker_color='blue'
    ))

    fig.add_trace(go.Bar(
        name='Benchmark',
        x=categories[:1],  # Only show return comparison
        y=benchmark_values[:1],
        marker_color='gray'
    ))

    fig.update_layout(
        title="Portfolio vs Benchmark Performance",
        yaxis_title="Value",
        barmode='group',
        height=400
    )

    return fig


def plot_rolling_attribution(rolling_attr):
    """Plot rolling attribution over time"""
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Returns Over Time', 'Attribution Effects'),
        vertical_spacing=0.12
    )

    # Returns
    fig.add_trace(
        go.Scatter(
            x=rolling_attr.index,
            y=rolling_attr['portfolio_return'] * 100,
            mode='lines',
            name='Portfolio',
            line=dict(color='blue')
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=rolling_attr.index,
            y=rolling_attr['benchmark_return'] * 100,
            mode='lines',
            name='Benchmark',
            line=dict(color='gray', dash='dash')
        ),
        row=1, col=1
    )

    # Attribution effects
    fig.add_trace(
        go.Scatter(
            x=rolling_attr.index,
            y=rolling_attr['allocation_effect'] * 100,
            mode='lines',
            name='Allocation',
            line=dict(color='green')
        ),
        row=2, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=rolling_attr.index,
            y=rolling_attr['selection_effect'] * 100,
            mode='lines',
            name='Selection',
            line=dict(color='orange')
        ),
        row=2, col=1
    )

    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Return (%)", row=1, col=1)
    fig.update_yaxes(title_text="Effect (%)", row=2, col=1)

    fig.update_layout(height=600, hovermode='x unified')

    return fig


def plot_factor_attribution(factor_attr):
    """Plot factor attribution"""
    factors = list(factor_attr.keys())
    contributions = [v * 100 for v in factor_attr.values()]

    colors = ['green' if v > 0 else 'red' for v in contributions]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=factors,
        y=contributions,
        marker_color=colors,
        text=[f"{v:+.2f}%" for v in contributions],
        textposition='outside'
    ))

    fig.update_layout(
        title="Factor Attribution (Annualized)",
        xaxis_title="Factor",
        yaxis_title="Contribution (%)",
        height=400
    )

    return fig


def plot_cumulative_returns(portfolio_returns, benchmark_returns):
    """Plot cumulative returns comparison"""
    portfolio_cum = (1 + portfolio_returns).cumprod() - 1
    benchmark_cum = (1 + benchmark_returns).cumprod() - 1

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=portfolio_cum.index,
        y=portfolio_cum * 100,
        mode='lines',
        name='Portfolio',
        line=dict(color='blue', width=2)
    ))

    fig.add_trace(go.Scatter(
        x=benchmark_cum.index,
        y=benchmark_cum * 100,
        mode='lines',
        name='Benchmark (Equal Weight)',
        line=dict(color='gray', width=2, dash='dash')
    ))

    # Add shaded area for outperformance
    fig.add_trace(go.Scatter(
        x=portfolio_cum.index,
        y=(portfolio_cum - benchmark_cum) * 100,
        mode='lines',
        name='Outperformance',
        fill='tozeroy',
        line=dict(color='green', width=1),
        opacity=0.3
    ))

    fig.update_layout(
        title="Cumulative Returns: Portfolio vs Benchmark",
        xaxis_title="Date",
        yaxis_title="Cumulative Return (%)",
        height=400,
        hovermode='x unified'
    )

    return fig


def main():
    st.set_page_config(
        page_title="Performance Attribution Dashboard",
        page_icon="📊",
        layout="wide"
    )

    st.title("📊 Performance Attribution Dashboard")
    st.markdown("**Analyze portfolio performance with Brinson attribution and factor analysis**")

    # Sidebar configuration
    st.sidebar.header("Portfolio Configuration")

    # Asset selection
    data_source = CryptoDataSource()
    available_symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'ADA/USDT']

    symbols = st.sidebar.multiselect(
        "Select Assets",
        available_symbols,
        default=['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
    )

    if len(symbols) < 2:
        st.warning("Please select at least 2 assets")
        return

    # Weights configuration
    st.sidebar.subheader("Portfolio Weights")
    weights = {}
    remaining = 1.0

    for i, symbol in enumerate(symbols):
        if i < len(symbols) - 1:
            weight = st.sidebar.slider(
                f"{symbol}",
                0.0, remaining, remaining / (len(symbols) - i),
                0.01,
                key=f"weight_{symbol}"
            )
            weights[symbol] = weight
            remaining -= weight
        else:
            weights[symbol] = remaining
            st.sidebar.text(f"{symbol}: {remaining:.2%}")

    # Time period
    days = st.sidebar.slider("Historical Data (days)", 30, 365, 90)

    # Attribution parameters
    st.sidebar.subheader("Attribution Parameters")
    risk_free_rate = st.sidebar.slider("Risk-Free Rate", 0.0, 0.1, 0.02, 0.01)

    # Benchmark option
    use_custom_benchmark = st.sidebar.checkbox("Use Custom Benchmark", value=False)

    benchmark_weights = None
    if use_custom_benchmark:
        st.sidebar.subheader("Benchmark Weights")
        benchmark_weights = {}
        remaining_bm = 1.0

        for i, symbol in enumerate(symbols):
            if i < len(symbols) - 1:
                bm_weight = st.sidebar.slider(
                    f"{symbol} (BM)",
                    0.0, remaining_bm, remaining_bm / (len(symbols) - i),
                    0.01,
                    key=f"bm_weight_{symbol}"
                )
                benchmark_weights[symbol] = bm_weight
                remaining_bm -= bm_weight
            else:
                benchmark_weights[symbol] = remaining_bm
                st.sidebar.text(f"{symbol} (BM): {remaining_bm:.2%}")

    # Load data button
    if st.sidebar.button("Load Data & Analyze", type="primary"):
        with st.spinner("Loading data and analyzing performance..."):
            try:
                # Fetch data
                data_frames = {}
                for symbol in symbols:
                    df = data_source.fetch_ohlcv(symbol, '1d', days=days)
                    data_frames[symbol] = df

                # Align all dataframes
                common_index = data_frames[symbols[0]].index
                for symbol in symbols[1:]:
                    common_index = common_index.intersection(data_frames[symbol].index)

                # Calculate returns
                returns_dict = {}
                for symbol in symbols:
                    df = data_frames[symbol]
                    returns = df['close'].pct_change().dropna()
                    returns_dict[symbol] = returns.loc[common_index]

                returns_df = pd.DataFrame(returns_dict)

                # Create Performance Attributor
                attributor = PerformanceAttributor(
                    returns_df,
                    weights,
                    benchmark_weights=benchmark_weights,
                    risk_free_rate=risk_free_rate
                )

                # Store in session state
                st.session_state['attributor'] = attributor
                st.session_state['returns_df'] = returns_df
                st.session_state['data_loaded'] = True

                st.success("✅ Data loaded successfully!")

            except Exception as e:
                st.error(f"Error loading data: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
                return

    # Main dashboard
    if st.session_state.get('data_loaded', False):
        attributor = st.session_state['attributor']
        returns_df = st.session_state['returns_df']

        # Tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Attribution Overview",
            "💼 Asset Analysis",
            "📈 Rolling Analysis",
            "🎯 Factor Attribution"
        ])

        # Tab 1: Attribution Overview
        with tab1:
            st.header("Performance Attribution Overview")

            # Calculate metrics
            brinson = attributor.brinson_attribution()
            metrics = attributor.calculate_performance_metrics()
            risk_attr = attributor.risk_attribution()

            # Key metrics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "Portfolio Return",
                    f"{brinson.total_return:.2%}",
                    help="Total portfolio return"
                )
                st.metric(
                    "Benchmark Return",
                    f"{brinson.benchmark_return:.2%}"
                )

            with col2:
                st.metric(
                    "Active Return",
                    f"{brinson.active_return:.2%}",
                    help="Portfolio - Benchmark"
                )
                st.metric(
                    "Information Ratio",
                    f"{metrics.information_ratio:.2f}"
                )

            with col3:
                st.metric(
                    "Sharpe Ratio",
                    f"{metrics.sharpe_ratio:.2f}"
                )
                st.metric(
                    "Sortino Ratio",
                    f"{metrics.sortino_ratio:.2f}"
                )

            with col4:
                st.metric(
                    "Max Drawdown",
                    f"{metrics.max_drawdown:.2%}"
                )
                st.metric(
                    "Calmar Ratio",
                    f"{metrics.calmar_ratio:.2f}"
                )

            # Brinson Attribution Chart
            st.subheader("Brinson Attribution Analysis")
            fig = plot_brinson_attribution(brinson)
            st.plotly_chart(fig, use_container_width=True)

            # Attribution effects summary
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Allocation Effect",
                    f"{brinson.allocation_effect:.2%}",
                    help="Returns from over/underweight decisions"
                )

            with col2:
                st.metric(
                    "Selection Effect",
                    f"{brinson.selection_effect:.2%}",
                    help="Returns from asset picking"
                )

            with col3:
                st.metric(
                    "Interaction Effect",
                    f"{brinson.interaction_effect:.2%}",
                    help="Combined allocation + selection"
                )

            # Cumulative returns comparison
            st.subheader("Cumulative Performance")
            fig = plot_cumulative_returns(
                attributor.portfolio_returns,
                attributor.benchmark_returns
            )
            st.plotly_chart(fig, use_container_width=True)

        # Tab 2: Asset Analysis
        with tab2:
            st.header("Asset-Level Analysis")

            # Asset contribution to return
            st.subheader("Asset Contribution to Active Return")
            fig = plot_asset_contribution(brinson.asset_contributions)
            st.plotly_chart(fig, use_container_width=True)

            # Detailed table
            st.subheader("Detailed Asset Attribution")
            asset_data = []
            for asset, contrib in brinson.asset_contributions.items():
                asset_data.append({
                    'Asset': asset,
                    'Allocation (%)': contrib['allocation'] * 100,
                    'Selection (%)': contrib['selection'] * 100,
                    'Interaction (%)': contrib['interaction'] * 100,
                    'Total (%)': contrib['total'] * 100
                })

            asset_df = pd.DataFrame(asset_data)
            st.dataframe(asset_df, use_container_width=True)

            # Risk attribution
            st.subheader("Risk Contribution by Asset")

            col1, col2 = st.columns(2)

            with col1:
                fig = plot_risk_contribution(risk_attr)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.metric("Portfolio Risk", f"{risk_attr.portfolio_risk:.2%}")
                st.metric(
                    "Diversification Ratio",
                    f"{risk_attr.diversification_ratio:.2f}",
                    help=">1 means diversification benefit"
                )

                # Risk contribution table
                risk_data = []
                for asset in risk_attr.risk_contribution_pct.keys():
                    risk_data.append({
                        'Asset': asset,
                        'Component Risk': risk_attr.component_risk[asset],
                        'Contribution (%)': risk_attr.risk_contribution_pct[asset] * 100
                    })

                risk_df = pd.DataFrame(risk_data)
                st.dataframe(risk_df, use_container_width=True)

        # Tab 3: Rolling Analysis
        with tab3:
            st.header("Rolling Attribution Analysis")

            window = st.slider("Rolling Window (days)", 10, 90, 30)

            with st.spinner("Calculating rolling attribution..."):
                rolling_attr = attributor.rolling_attribution(window=window)

            fig = plot_rolling_attribution(rolling_attr)
            st.plotly_chart(fig, use_container_width=True)

            # Statistics
            st.subheader("Rolling Statistics")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Avg Active Return",
                    f"{rolling_attr['active_return'].mean():.2%}"
                )

            with col2:
                st.metric(
                    "Avg Allocation Effect",
                    f"{rolling_attr['allocation_effect'].mean():.2%}"
                )

            with col3:
                st.metric(
                    "Avg Selection Effect",
                    f"{rolling_attr['selection_effect'].mean():.2%}"
                )

        # Tab 4: Factor Attribution
        with tab4:
            st.header("Factor Attribution Analysis")

            st.markdown("""
            Factor attribution decomposes returns into common factors:
            - **Market**: Overall market movement
            - **Value**: Value spread between assets
            - **Momentum**: Momentum effect
            - **Alpha**: Unexplained (skill-based) returns
            """)

            factor_attr = attributor.factor_attribution()

            fig = plot_factor_attribution(factor_attr)
            st.plotly_chart(fig, use_container_width=True)

            # Factor table
            st.subheader("Factor Contributions")
            factor_data = []
            for factor, contrib in factor_attr.items():
                factor_data.append({
                    'Factor': factor,
                    'Contribution (%)': contrib * 100
                })

            factor_df = pd.DataFrame(factor_data)
            st.dataframe(factor_df, use_container_width=True)

            # Time-weighted vs Money-weighted returns
            st.subheader("Return Calculation Methods")
            col1, col2 = st.columns(2)

            with col1:
                twr = attributor.time_weighted_return()
                st.metric(
                    "Time-Weighted Return",
                    f"{twr:.2%}",
                    help="Geometric average, independent of cash flows"
                )

            with col2:
                mwr = attributor.money_weighted_return()
                st.metric(
                    "Money-Weighted Return",
                    f"{mwr:.2%}",
                    help="IRR, accounts for timing of cash flows"
                )

            # Full report
            st.subheader("📄 Full Attribution Report")
            with st.expander("View Complete Text Report"):
                report = generate_attribution_report(attributor)
                st.code(report)

    else:
        st.info("👈 Configure portfolio in sidebar and click 'Load Data & Analyze'")


if __name__ == "__main__":
    main()
