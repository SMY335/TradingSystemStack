"""
Risk Analysis Dashboard

Interactive Streamlit dashboard for portfolio risk analysis and visualization.
Features:
- Risk metrics display
- VaR and CVaR visualization
- Stress test results
- Monte Carlo simulation
- Correlation heatmap
- Historical drawdown chart
- Scenario analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.portfolio.risk_manager import RiskManager, RiskMetrics


class RiskDashboard:
    """Interactive risk analysis dashboard"""
    
    def __init__(self):
        """Initialize dashboard"""
        st.set_page_config(
            page_title="Portfolio Risk Analysis",
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
    def run(self):
        """Run the dashboard"""
        st.title("📊 Portfolio Risk Analysis Dashboard")
        st.markdown("---")
        
        # Sidebar for data input
        self._render_sidebar()
        
        # Main tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "📈 Risk Analysis",
            "🎲 Monte Carlo",
            "🔥 Stress Testing",
            "🔗 Scenario Analysis"
        ])
        
        with tab1:
            self._render_risk_analysis_tab()
        
        with tab2:
            self._render_monte_carlo_tab()
        
        with tab3:
            self._render_stress_testing_tab()
        
        with tab4:
            self._render_scenario_analysis_tab()
    
    def _render_sidebar(self):
        """Render sidebar with data input options"""
        st.sidebar.header("⚙️ Configuration")
        
        # Data source selection
        data_source = st.sidebar.radio(
            "Data Source",
            ["Sample Data", "Upload CSV", "Live Data (API)"]
        )
        
        if data_source == "Sample Data":
            self._load_sample_data()
        elif data_source == "Upload CSV":
            self._load_uploaded_data()
        else:
            st.sidebar.info("Live API integration coming soon!")
            self._load_sample_data()
        
        st.sidebar.markdown("---")
        
        # Portfolio weights configuration
        st.sidebar.subheader("Portfolio Weights")
        
        if hasattr(st.session_state, 'returns') and st.session_state.returns is not None:
            assets = st.session_state.returns.columns.tolist()
            weights = {}
            
            weight_method = st.sidebar.radio(
                "Weight Method",
                ["Equal Weight", "Custom"]
            )
            
            if weight_method == "Equal Weight":
                for asset in assets:
                    weights[asset] = 1.0 / len(assets)
            else:
                remaining = 100.0
                for i, asset in enumerate(assets):
                    if i < len(assets) - 1:
                        weight = st.sidebar.slider(
                            f"{asset} (%)",
                            0.0, 100.0, 100.0 / len(assets),
                            key=f"weight_{asset}"
                        )
                        weights[asset] = weight / 100.0
                        remaining -= weight
                    else:
                        weights[asset] = max(0, remaining) / 100.0
                        st.sidebar.info(f"{asset}: {remaining:.1f}%")
            
            st.session_state.weights = weights
            
            # Display total
            total = sum(weights.values()) * 100
            st.sidebar.metric("Total Weight", f"{total:.1f}%")
        
        st.sidebar.markdown("---")
        
        # Risk parameters
        st.sidebar.subheader("Risk Parameters")
        st.session_state.confidence_95 = st.sidebar.slider(
            "Confidence Level (VaR 95%)",
            0.90, 0.99, 0.95, 0.01
        )
        st.session_state.confidence_99 = st.sidebar.slider(
            "Confidence Level (VaR 99%)",
            0.95, 0.999, 0.99, 0.001
        )
        st.session_state.risk_free_rate = st.sidebar.number_input(
            "Risk-Free Rate (%)",
            0.0, 10.0, 2.0, 0.1
        ) / 100.0
    
    def _load_sample_data(self):
        """Load sample data for demonstration"""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', '2024-12-31', freq='D')
        
        # Simulate crypto-like returns
        returns_data = {
            'BTC': np.random.normal(0.001, 0.04, len(dates)),
            'ETH': np.random.normal(0.0012, 0.045, len(dates)),
            'SOL': np.random.normal(0.0015, 0.055, len(dates)),
            'AVAX': np.random.normal(0.001, 0.05, len(dates)),
            'MATIC': np.random.normal(0.0008, 0.048, len(dates)),
        }
        
        st.session_state.returns = pd.DataFrame(returns_data, index=dates)
        st.sidebar.success("✅ Sample data loaded (5 assets, 2020-2024)")
    
    def _load_uploaded_data(self):
        """Load data from uploaded CSV"""
        uploaded_file = st.sidebar.file_uploader(
            "Upload Returns CSV",
            type=['csv'],
            help="CSV with dates as index and asset returns as columns"
        )
        
        if uploaded_file is not None:
            try:
                returns = pd.read_csv(uploaded_file, index_col=0, parse_dates=True)
                st.session_state.returns = returns
                st.sidebar.success(f"✅ Data loaded: {len(returns.columns)} assets, {len(returns)} days")
            except Exception as e:
                st.sidebar.error(f"Error loading file: {str(e)}")
        else:
            st.sidebar.info("Please upload a CSV file")
            # Load sample data as fallback
            self._load_sample_data()
    
    def _get_risk_manager(self) -> Optional[RiskManager]:
        """Get RiskManager instance"""
        if not hasattr(st.session_state, 'returns') or st.session_state.returns is None:
            return None
        
        weights = getattr(st.session_state, 'weights', None)
        risk_free_rate = getattr(st.session_state, 'risk_free_rate', 0.02)
        
        return RiskManager(
            st.session_state.returns,
            weights,
            risk_free_rate=risk_free_rate
        )
    
    def _render_risk_analysis_tab(self):
        """Render main risk analysis tab"""
        st.header("Risk Metrics Analysis")
        
        risk_manager = self._get_risk_manager()
        if risk_manager is None:
            st.warning("Please configure data in the sidebar")
            return
        
        # Calculate metrics
        metrics = risk_manager.calculate_risk_metrics()
        
        # Display key metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("VaR 95%", f"{metrics.var_95:.2%}", delta=None)
            st.metric("CVaR 95%", f"{metrics.cvar_95:.2%}", delta=None)
        
        with col2:
            st.metric("VaR 99%", f"{metrics.var_99:.2%}", delta=None)
            st.metric("CVaR 99%", f"{metrics.cvar_99:.2%}", delta=None)
        
        with col3:
            st.metric("Volatility (ann.)", f"{metrics.volatility:.2%}")
            st.metric("Max Drawdown", f"{metrics.max_drawdown:.2%}")
        
        with col4:
            st.metric("Sharpe Ratio", f"{metrics.sharpe_ratio:.2f}")
            st.metric("Sortino Ratio", f"{metrics.sortino_ratio:.2f}")
        
        st.markdown("---")
        
        # VaR comparison chart
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("VaR & CVaR Comparison")
            fig = self._create_var_comparison_chart(metrics)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Distribution Analysis")
            fig = self._create_distribution_chart(risk_manager)
            st.plotly_chart(fig, use_container_width=True)
        
        # Correlation heatmap
        st.subheader("Asset Correlation Heatmap")
        fig = self._create_correlation_heatmap(risk_manager)
        st.plotly_chart(fig, use_container_width=True)
        
        # Drawdown chart
        st.subheader("Historical Drawdown")
        fig = self._create_drawdown_chart(risk_manager)
        st.plotly_chart(fig, use_container_width=True)
        
        # Tail risk metrics
        st.subheader("Tail Risk Metrics")
        tail_metrics = risk_manager.calculate_tail_risk_metrics()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Skewness", f"{tail_metrics['skewness']:.3f}")
        with col2:
            st.metric("Kurtosis", f"{tail_metrics['kurtosis']:.3f}")
        with col3:
            st.metric("Worst 1%", f"{tail_metrics['worst_1pct']:.2%}")
        with col4:
            st.metric("Best 99%", f"{tail_metrics['best_99pct']:.2%}")
        
        # Risk alerts
        st.subheader("⚠️ Risk Alerts")
        alerts = risk_manager.check_risk_alerts()
        
        if any(alerts['critical']):
            for alert in alerts['critical']:
                st.error(f"🚨 CRITICAL: {alert}")
        
        if any(alerts['warning']):
            for alert in alerts['warning']:
                st.warning(f"⚠️ WARNING: {alert}")
        
        if not any(alerts['critical']) and not any(alerts['warning']):
            st.success("✅ All risk metrics within acceptable ranges")
    
    def _render_monte_carlo_tab(self):
        """Render Monte Carlo simulation tab"""
        st.header("Monte Carlo Simulation")
        
        risk_manager = self._get_risk_manager()
        if risk_manager is None:
            st.warning("Please configure data in the sidebar")
            return
        
        # Simulation parameters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            n_simulations = st.number_input(
                "Number of Simulations",
                1000, 50000, 10000, 1000
            )
        
        with col2:
            time_horizon = st.number_input(
                "Time Horizon (days)",
                1, 1000, 252, 1
            )
        
        with col3:
            initial_value = st.number_input(
                "Initial Portfolio Value ($)",
                10000, 10000000, 1000000, 10000
            )
        
        if st.button("Run Simulation", type="primary"):
            with st.spinner("Running Monte Carlo simulation..."):
                final_values, paths = risk_manager.monte_carlo_simulation(
                    n_simulations=n_simulations,
                    time_horizon=time_horizon,
                    initial_value=initial_value
                )
                
                st.session_state.mc_final_values = final_values
                st.session_state.mc_paths = paths
        
        if hasattr(st.session_state, 'mc_final_values'):
            final_values = st.session_state.mc_final_values
            
            # Summary statistics
            st.subheader("Simulation Results")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Mean", f"${final_values.mean():,.0f}")
            with col2:
                st.metric("Median", f"${np.median(final_values):,.0f}")
            with col3:
                st.metric("Std Dev", f"${final_values.std():,.0f}")
            with col4:
                var_5 = np.percentile(final_values, 5)
                st.metric("VaR 95%", f"${var_5:,.0f}")
            
            # Distribution histogram
            st.subheader("Final Value Distribution")
            fig = go.Figure()
            
            fig.add_trace(go.Histogram(
                x=final_values,
                nbinsx=50,
                name='Final Values',
                marker_color='lightblue'
            ))
            
            # Add percentile lines
            percentiles = [5, 50, 95]
            colors = ['red', 'green', 'blue']
            for p, color in zip(percentiles, colors):
                val = np.percentile(final_values, p)
                fig.add_vline(
                    x=val,
                    line_dash="dash",
                    line_color=color,
                    annotation_text=f"{p}th: ${val:,.0f}"
                )
            
            fig.update_layout(
                xaxis_title="Final Portfolio Value ($)",
                yaxis_title="Frequency",
                showlegend=False,
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Sample paths
            if hasattr(st.session_state, 'mc_paths'):
                st.subheader("Sample Simulation Paths")
                fig = go.Figure()
                
                paths = st.session_state.mc_paths
                for col in paths.columns[:20]:  # Show first 20 paths
                    fig.add_trace(go.Scatter(
                        y=paths[col],
                        mode='lines',
                        line=dict(width=1),
                        opacity=0.3,
                        showlegend=False
                    ))
                
                fig.update_layout(
                    xaxis_title="Time (days)",
                    yaxis_title="Portfolio Value ($)",
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)
    
    def _render_stress_testing_tab(self):
        """Render stress testing tab"""
        st.header("Stress Testing")
        
        risk_manager = self._get_risk_manager()
        if risk_manager is None:
            st.warning("Please configure data in the sidebar")
            return
        
        # Run stress tests
        stress_results = risk_manager.stress_test()
        
        # Display results in a table
        st.subheader("Stress Test Scenarios")
        
        results_data = []
        for result in stress_results:
            results_data.append({
                'Scenario': result.scenario_name,
                'Portfolio Loss': f"{result.loss_percentage:.2f}%",
                'Loss Value': f"${result.portfolio_loss * 1000000:,.0f}",
                'VaR Breach': '🚨' if result.var_breach else '✅'
            })
        
        df = pd.DataFrame(results_data)
        st.dataframe(df, use_container_width=True)
        
        # Visualization
        st.subheader("Stress Test Impact")
        fig = go.Figure()
        
        scenarios = [r.scenario_name for r in stress_results]
        losses = [r.loss_percentage for r in stress_results]
        colors = ['red' if r.var_breach else 'orange' for r in stress_results]
        
        fig.add_trace(go.Bar(
            x=scenarios,
            y=losses,
            marker_color=colors,
            text=[f"{l:.1f}%" for l in losses],
            textposition='outside'
        ))
        
        fig.update_layout(
            xaxis_title="Scenario",
            yaxis_title="Portfolio Loss (%)",
            height=500,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Asset-level impact for selected scenario
        st.subheader("Asset-Level Impact")
        selected_scenario = st.selectbox(
            "Select Scenario",
            scenarios
        )
        
        result = next(r for r in stress_results if r.scenario_name == selected_scenario)
        
        asset_data = pd.DataFrame([
            {'Asset': asset, 'Impact (%)': impact * 100}
            for asset, impact in result.asset_impacts.items()
        ]).sort_values('Impact (%)')
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=asset_data['Impact (%)'],
            y=asset_data['Asset'],
            orientation='h',
            marker_color=['red' if x < 0 else 'green' for x in asset_data['Impact (%)']]
        ))
        
        fig.update_layout(
            xaxis_title="Impact (%)",
            yaxis_title="Asset",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_scenario_analysis_tab(self):
        """Render custom scenario analysis tab"""
        st.header("Custom Scenario Analysis")
        
        risk_manager = self._get_risk_manager()
        if risk_manager is None:
            st.warning("Please configure data in the sidebar")
            return
        
        st.info("Create custom scenarios to test portfolio resilience")
        
        # Scenario builder
        st.subheader("Scenario Builder")
        
        scenario_name = st.text_input("Scenario Name", "Custom Scenario")
        
        st.write("Asset Shocks (%):")
        
        assets = risk_manager.assets
        shocks = {}
        
        cols = st.columns(min(3, len(assets)))
        for i, asset in enumerate(assets):
            with cols[i % len(cols)]:
                shock = st.slider(
                    asset,
                    -100.0, 100.0, 0.0, 1.0,
                    key=f"shock_{asset}"
                )
                shocks[asset] = shock / 100.0
        
        if st.button("Run Custom Scenario", type="primary"):
            scenarios = {scenario_name: shocks}
            results = risk_manager.stress_test(scenarios)
            
            if results:
                result = results[0]
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        "Portfolio Impact",
                        f"{result.loss_percentage:.2f}%",
                        delta=f"${result.portfolio_loss * 1000000:,.0f}"
                    )
                with col2:
                    breach_status = "🚨 VaR Breach" if result.var_breach else "✅ Within VaR"
                    st.metric("VaR Status", breach_status)
                
                # Asset breakdown
                st.subheader("Asset Contribution")
                asset_data = pd.DataFrame([
                    {'Asset': asset, 'Contribution (%)': impact * 100}
                    for asset, impact in result.asset_impacts.items()
                ])
                
                fig = px.pie(
                    asset_data,
                    values=asset_data['Contribution (%)'].abs(),
                    names='Asset',
                    title='Impact Contribution by Asset'
                )
                st.plotly_chart(fig, use_container_width=True)
    
    def _create_var_comparison_chart(self, metrics: RiskMetrics):
        """Create VaR vs CVaR comparison chart"""
        fig = go.Figure()
        
        categories = ['VaR 95%', 'CVaR 95%', 'VaR 99%', 'CVaR 99%']
        values = [
            metrics.var_95 * 100,
            metrics.cvar_95 * 100,
            metrics.var_99 * 100,
            metrics.cvar_99 * 100
        ]
        
        fig.add_trace(go.Bar(
            x=categories,
            y=values,
            marker_color=['lightblue', 'darkblue', 'lightcoral', 'darkred'],
            text=[f"{v:.2f}%" for v in values],
            textposition='outside'
        ))
        
        fig.update_layout(
            yaxis_title="Value (%)",
            showlegend=False,
            height=400
        )
        
        return fig
    
    def _create_distribution_chart(self, risk_manager: RiskManager):
        """Create returns distribution chart"""
        returns = risk_manager.portfolio_returns.dropna()
        
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=returns * 100,
            nbinsx=50,
            name='Returns',
            marker_color='lightblue'
        ))
        
        # Add VaR line
        var_95 = risk_manager.calculate_var_historical(0.95)
        fig.add_vline(
            x=-var_95 * 100,
            line_dash="dash",
            line_color="red",
            annotation_text="VaR 95%"
        )
        
        fig.update_layout(
            xaxis_title="Daily Returns (%)",
            yaxis_title="Frequency",
            showlegend=False,
            height=400
        )
        
        return fig
    
    def _create_correlation_heatmap(self, risk_manager: RiskManager):
        """Create correlation heatmap"""
        corr_matrix = risk_manager.correlation_matrix
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=corr_matrix.values,
            texttemplate='%{text:.2f}',
            textfont={"size": 10},
            colorbar=dict(title="Correlation")
        ))
        
        fig.update_layout(
            height=500,
            xaxis_title="Asset",
            yaxis_title="Asset"
        )
        
        return fig
    
    def _create_drawdown_chart(self, risk_manager: RiskManager):
        """Create drawdown chart"""
        _, drawdown = risk_manager.calculate_max_drawdown()
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=drawdown.index,
            y=drawdown * 100,
            fill='tozeroy',
            line_color='red',
            name='Drawdown'
        ))
        
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Drawdown (%)",
            showlegend=False,
            height=400
        )
        
        return fig


def main():
    """Main function to run dashboard"""
    dashboard = RiskDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()
