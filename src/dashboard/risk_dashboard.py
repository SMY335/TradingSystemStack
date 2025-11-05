"""
Risk Management Dashboard
Interactive Streamlit dashboard for portfolio risk analysis
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

from src.portfolio.risk_manager import RiskManager, generate_risk_report
from src.data_sources.crypto_data import CryptoDataSource


def plot_var_distribution(portfolio_returns: pd.Series, var_95: float, var_99: float, cvar_95: float):
    """Plot returns distribution with VaR and CVaR"""
    fig = go.Figure()

    # Histogram of returns
    fig.add_trace(go.Histogram(
        x=portfolio_returns * 100,
        nbinsx=50,
        name='Returns Distribution',
        opacity=0.7
    ))

    # VaR lines
    fig.add_vline(
        x=var_95 * 100,
        line_dash="dash",
        line_color="orange",
        annotation_text=f"VaR 95%: {var_95:.2%}",
        annotation_position="top"
    )

    fig.add_vline(
        x=var_99 * 100,
        line_dash="dash",
        line_color="red",
        annotation_text=f"VaR 99%: {var_99:.2%}",
        annotation_position="bottom"
    )

    fig.add_vline(
        x=cvar_95 * 100,
        line_dash="dot",
        line_color="darkred",
        annotation_text=f"CVaR 95%: {cvar_95:.2%}",
        annotation_position="top"
    )

    fig.update_layout(
        title="Portfolio Returns Distribution with VaR/CVaR",
        xaxis_title="Daily Return (%)",
        yaxis_title="Frequency",
        showlegend=True,
        height=400
    )

    return fig


def plot_rolling_var(rolling_var: pd.Series, portfolio_returns: pd.Series):
    """Plot rolling VaR over time"""
    fig = go.Figure()

    # Rolling VaR
    fig.add_trace(go.Scatter(
        x=rolling_var.index,
        y=rolling_var * 100,
        mode='lines',
        name='Rolling VaR 95%',
        line=dict(color='red')
    ))

    # Actual returns
    fig.add_trace(go.Scatter(
        x=portfolio_returns.index,
        y=portfolio_returns * 100,
        mode='lines',
        name='Portfolio Returns',
        line=dict(color='lightblue', width=1),
        opacity=0.5
    ))

    fig.update_layout(
        title="Rolling VaR (30-day window)",
        xaxis_title="Date",
        yaxis_title="Return (%)",
        height=400,
        hovermode='x unified'
    )

    return fig


def plot_stress_test_results(stress_results):
    """Plot stress test scenarios"""
    scenarios = [s.scenario_name for s in stress_results]
    losses = [s.portfolio_loss for s in stress_results]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=scenarios,
        y=losses,
        marker_color=['red' if l < 0 else 'green' for l in losses],
        text=[f"{l:.1f}%" for l in losses],
        textposition='outside'
    ))

    fig.update_layout(
        title="Stress Test Results - Portfolio Impact",
        xaxis_title="Scenario",
        yaxis_title="Portfolio Loss (%)",
        height=400
    )

    return fig


def plot_correlation_heatmap(correlation_matrix: pd.DataFrame):
    """Plot correlation heatmap"""
    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix.values,
        x=correlation_matrix.columns,
        y=correlation_matrix.index,
        colorscale='RdBu',
        zmid=0,
        text=correlation_matrix.values,
        texttemplate='%{text:.2f}',
        textfont={"size": 10},
        colorbar=dict(title="Correlation")
    ))

    fig.update_layout(
        title="Asset Correlation Matrix",
        height=400
    )

    return fig


def plot_component_var(component_var: dict):
    """Plot Component VaR contributions"""
    assets = list(component_var.keys())
    values = list(component_var.values())

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=assets,
        y=values,
        marker_color=['red' if v > 0 else 'green' for v in values],
        text=[f"{v:.4f}" for v in values],
        textposition='outside'
    ))

    fig.update_layout(
        title="Component VaR - Risk Contribution by Asset",
        xaxis_title="Asset",
        yaxis_title="Component VaR",
        height=400
    )

    return fig


def plot_monte_carlo_simulation(risk_manager: RiskManager):
    """Plot Monte Carlo simulation results"""
    simulated_returns = risk_manager.monte_carlo_simulation(n_scenarios=10000)

    fig = go.Figure()

    # Histogram
    fig.add_trace(go.Histogram(
        x=simulated_returns * 100,
        nbinsx=50,
        name='Simulated Returns',
        opacity=0.7
    ))

    # Add percentiles
    p5 = np.percentile(simulated_returns, 5) * 100
    p95 = np.percentile(simulated_returns, 95) * 100

    fig.add_vline(
        x=p5,
        line_dash="dash",
        line_color="red",
        annotation_text=f"5th percentile: {p5:.2f}%"
    )

    fig.add_vline(
        x=p95,
        line_dash="dash",
        line_color="green",
        annotation_text=f"95th percentile: {p95:.2f}%"
    )

    fig.update_layout(
        title="Monte Carlo Simulation (10,000 scenarios)",
        xaxis_title="Simulated Return (%)",
        yaxis_title="Frequency",
        height=400
    )

    return fig


def plot_drawdown(portfolio_returns: pd.Series):
    """Plot drawdown chart"""
    cumulative = (1 + portfolio_returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max

    fig = go.Figure()

    # Cumulative returns
    fig.add_trace(go.Scatter(
        x=cumulative.index,
        y=(cumulative - 1) * 100,
        mode='lines',
        name='Cumulative Return',
        line=dict(color='blue')
    ))

    # Drawdown (filled area)
    fig.add_trace(go.Scatter(
        x=drawdown.index,
        y=drawdown * 100,
        mode='lines',
        name='Drawdown',
        fill='tozeroy',
        line=dict(color='red'),
        yaxis='y2'
    ))

    fig.update_layout(
        title="Cumulative Returns and Drawdown",
        xaxis_title="Date",
        yaxis=dict(title="Cumulative Return (%)", side='left'),
        yaxis2=dict(title="Drawdown (%)", side='right', overlaying='y'),
        height=400,
        hovermode='x unified'
    )

    return fig


def main():
    st.set_page_config(
        page_title="Risk Management Dashboard",
        page_icon="⚠️",
        layout="wide"
    )

    st.title("⚠️ Risk Management Dashboard")
    st.markdown("**Advanced portfolio risk analysis and stress testing**")

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

    # Risk parameters
    st.sidebar.subheader("Risk Parameters")
    risk_free_rate = st.sidebar.slider("Risk-Free Rate", 0.0, 0.1, 0.02, 0.01)

    # Load data button
    if st.sidebar.button("Load Data & Calculate Risk", type="primary"):
        with st.spinner("Loading data and calculating risk metrics..."):
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

                # Create Risk Manager
                risk_manager = RiskManager(
                    returns_df,
                    weights,
                    risk_free_rate=risk_free_rate
                )

                # Store in session state
                st.session_state['risk_manager'] = risk_manager
                st.session_state['returns_df'] = returns_df
                st.session_state['data_loaded'] = True

                st.success("✅ Data loaded successfully!")

            except Exception as e:
                st.error(f"Error loading data: {str(e)}")
                return

    # Main dashboard
    if st.session_state.get('data_loaded', False):
        risk_manager = st.session_state['risk_manager']
        returns_df = st.session_state['returns_df']

        # Tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Risk Overview",
            "🎲 Monte Carlo",
            "💥 Stress Tests",
            "📈 Advanced Metrics"
        ])

        # Tab 1: Risk Overview
        with tab1:
            st.header("Risk Metrics Overview")

            # Calculate metrics
            metrics = risk_manager.calculate_risk_metrics()
            summary = risk_manager.get_risk_summary()

            # Display key metrics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("VaR 95%", f"{metrics.var_95:.2%}", help="Maximum loss exceeded 5% of time")
                st.metric("CVaR 95%", f"{metrics.cvar_95:.2%}", help="Expected loss when VaR exceeded")

            with col2:
                st.metric("VaR 99%", f"{metrics.var_99:.2%}", help="Maximum loss exceeded 1% of time")
                st.metric("CVaR 99%", f"{metrics.cvar_99:.2%}")

            with col3:
                st.metric("Volatility", f"{metrics.volatility:.2%}", help="Annualized volatility")
                st.metric("Max Drawdown", f"{metrics.max_drawdown:.2%}")

            with col4:
                st.metric("Sharpe Ratio", f"{metrics.sharpe_ratio:.2f}")
                st.metric("Sortino Ratio", f"{metrics.sortino_ratio:.2f}")

            # Charts
            col1, col2 = st.columns(2)

            with col1:
                fig = plot_var_distribution(
                    risk_manager.portfolio_returns,
                    metrics.var_95,
                    metrics.var_99,
                    metrics.cvar_95
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                rolling_var = risk_manager.rolling_var(window=30)
                fig = plot_rolling_var(rolling_var, risk_manager.portfolio_returns)
                st.plotly_chart(fig, use_container_width=True)

            # Drawdown chart
            fig = plot_drawdown(risk_manager.portfolio_returns)
            st.plotly_chart(fig, use_container_width=True)

            # Component VaR
            st.subheader("Risk Contribution by Asset")
            component_var = risk_manager.component_var()
            fig = plot_component_var(component_var)
            st.plotly_chart(fig, use_container_width=True)

        # Tab 2: Monte Carlo
        with tab2:
            st.header("Monte Carlo Simulation")

            st.markdown("""
            Monte Carlo simulation generates 10,000 random scenarios based on:
            - Historical mean returns
            - Historical covariance structure
            - Correlation between assets
            """)

            fig = plot_monte_carlo_simulation(risk_manager)
            st.plotly_chart(fig, use_container_width=True)

            # Statistics
            simulated = risk_manager.monte_carlo_simulation(10000)
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Mean Return", f"{simulated.mean():.2%}")
                st.metric("Median Return", f"{np.median(simulated):.2%}")

            with col2:
                st.metric("Std Deviation", f"{simulated.std():.2%}")
                st.metric("VaR 95% (MC)", f"{np.percentile(simulated, 5):.2%}")

            with col3:
                st.metric("Best Case (95%)", f"{np.percentile(simulated, 95):.2%}")
                st.metric("Worst Case (5%)", f"{np.percentile(simulated, 5):.2%}")

        # Tab 3: Stress Tests
        with tab3:
            st.header("Stress Test Scenarios")

            stress_results = risk_manager.stress_test()

            # Chart
            fig = plot_stress_test_results(stress_results)
            st.plotly_chart(fig, use_container_width=True)

            # Detailed results
            st.subheader("Detailed Stress Test Results")

            for result in stress_results:
                with st.expander(f"📉 {result.scenario_name}"):
                    col1, col2 = st.columns(2)

                    with col1:
                        st.metric("Portfolio Loss", f"{result.portfolio_loss:.2f}%")
                        st.metric("Worst Asset", result.worst_asset)

                    with col2:
                        st.metric("Worst Asset Loss", f"{result.worst_asset_loss:.2f}%")
                        st.metric("Diversification Benefit", f"{result.diversification_benefit:.2f}%")

        # Tab 4: Advanced Metrics
        with tab4:
            st.header("Advanced Risk Metrics")

            # Correlation matrix
            st.subheader("Asset Correlation Matrix")
            correlation = risk_manager.correlation_analysis()
            fig = plot_correlation_heatmap(correlation)
            st.plotly_chart(fig, use_container_width=True)

            # Marginal VaR
            st.subheader("Marginal VaR")
            st.markdown("*Change in portfolio VaR from a small increase in each position*")
            marginal_var = risk_manager.marginal_var()

            marginal_df = pd.DataFrame({
                'Asset': list(marginal_var.keys()),
                'Marginal VaR': list(marginal_var.values())
            })
            st.dataframe(marginal_df, use_container_width=True)

            # Tail risk metrics
            st.subheader("Tail Risk Metrics")
            col1, col2 = st.columns(2)

            with col1:
                st.metric("Skewness", f"{metrics.skewness:.3f}")
                st.caption("Negative = more downside risk")

            with col2:
                st.metric("Kurtosis", f"{metrics.kurtosis:.3f}")
                st.caption("Positive = fatter tails (more extreme events)")

            # Text report
            st.subheader("📄 Risk Report")
            with st.expander("View Full Text Report"):
                report = generate_risk_report(risk_manager)
                st.code(report)

    else:
        st.info("👈 Configure portfolio in sidebar and click 'Load Data & Calculate Risk'")


if __name__ == "__main__":
    main()
