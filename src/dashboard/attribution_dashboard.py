"""
Performance Attribution Dashboard

Interactive Streamlit dashboard for performance attribution analysis.
Features:
- Brinson attribution visualization
- Asset contribution waterfall charts
- Factor returns breakdown
- Rolling attribution analysis
- Risk attribution charts
- Performance metrics display
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

from src.portfolio.performance_attribution import (
    PerformanceAttributor,
    AttributionResult,
    RiskAttribution
)


class AttributionDashboard:
    """Interactive performance attribution dashboard"""
    
    def __init__(self):
        """Initialize dashboard"""
        st.set_page_config(
            page_title="Performance Attribution Analysis",
            page_icon="📈",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    def run(self):
        """Run the dashboard"""
        st.title("📈 Performance Attribution Dashboard")
        st.markdown("---")
        
        # Sidebar configuration
        self._render_sidebar()
        
        # Main tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Performance Attribution",
            "🎯 Risk Attribution",
            "📉 Rolling Analysis",
            "📝 Reports"
        ])
        
        with tab1:
            self._render_performance_attribution_tab()
        
        with tab2:
            self._render_risk_attribution_tab()
        
        with tab3:
            self._render_rolling_analysis_tab()
        
        with tab4:
            self._render_reports_tab()
    
    def _render_sidebar(self):
        """Render sidebar configuration"""
        st.sidebar.header("⚙️ Configuration")
        
        # Data source
        data_source = st.sidebar.radio(
            "Data Source",
            ["Sample Data", "Upload CSV"]
        )
        
        if data_source == "Sample Data":
            self._load_sample_data()
        else:
            self._load_uploaded_data()
        
        st.sidebar.markdown("---")
        
        # Portfolio weights
        if hasattr(st.session_state, 'returns') and st.session_state.returns is not None:
            st.sidebar.subheader("Portfolio Configuration")
            
            assets = st.session_state.returns.columns.tolist()
            
            # Portfolio weights
            st.sidebar.write("**Portfolio Weights:**")
            portfolio_weights = {}
            remaining = 100.0
            
            for i, asset in enumerate(assets):
                if i < len(assets) - 1:
                    weight = st.sidebar.slider(
                        f"{asset} (%)",
                        0.0, 100.0, 100.0 / len(assets),
                        key=f"port_{asset}"
                    )
                    portfolio_weights[asset] = weight / 100.0
                    remaining -= weight
                else:
                    portfolio_weights[asset] = max(0, remaining) / 100.0
                    st.sidebar.info(f"{asset}: {remaining:.1f}%")
            
            st.session_state.portfolio_weights = portfolio_weights
            
            # Benchmark weights
            st.sidebar.write("**Benchmark Weights:**")
            benchmark_option = st.sidebar.radio(
                "Benchmark Type",
                ["Equal Weight", "Custom"]
            )
            
            if benchmark_option == "Equal Weight":
                benchmark_weights = {asset: 1.0 / len(assets) for asset in assets}
            else:
                benchmark_weights = {}
                remaining = 100.0
                for i, asset in enumerate(assets):
                    if i < len(assets) - 1:
                        weight = st.sidebar.slider(
                            f"{asset} (%)",
                            0.0, 100.0, 100.0 / len(assets),
                            key=f"bench_{asset}"
                        )
                        benchmark_weights[asset] = weight / 100.0
                        remaining -= weight
                    else:
                        benchmark_weights[asset] = max(0, remaining) / 100.0
                        st.sidebar.info(f"{asset}: {remaining:.1f}%")
            
            st.session_state.benchmark_weights = benchmark_weights
    
    def _load_sample_data(self):
        """Load sample data"""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', '2024-12-31', freq='D')
        
        returns_data = {
            'BTC': np.random.normal(0.001, 0.04, len(dates)),
            'ETH': np.random.normal(0.0012, 0.045, len(dates)),
            'SOL': np.random.normal(0.0015, 0.055, len(dates)),
            'AVAX': np.random.normal(0.001, 0.05, len(dates)),
            'MATIC': np.random.normal(0.0008, 0.048, len(dates)),
        }
        
        st.session_state.returns = pd.DataFrame(returns_data, index=dates)
        st.sidebar.success("✅ Sample data loaded")
    
    def _load_uploaded_data(self):
        """Load uploaded CSV data"""
        uploaded_file = st.sidebar.file_uploader(
            "Upload Returns CSV",
            type=['csv']
        )
        
        if uploaded_file is not None:
            try:
                returns = pd.read_csv(uploaded_file, index_col=0, parse_dates=True)
                st.session_state.returns = returns
                st.sidebar.success(f"✅ Data loaded: {len(returns.columns)} assets")
            except Exception as e:
                st.sidebar.error(f"Error: {str(e)}")
        else:
            self._load_sample_data()
    
    def _get_attributor(self) -> Optional[PerformanceAttributor]:
        """Get PerformanceAttributor instance"""
        if not hasattr(st.session_state, 'returns'):
            return None
        
        portfolio_weights = getattr(st.session_state, 'portfolio_weights', None)
        benchmark_weights = getattr(st.session_state, 'benchmark_weights', None)
        
        if portfolio_weights is None:
            return None
        
        return PerformanceAttributor(
            st.session_state.returns,
            portfolio_weights,
            benchmark_weights
        )
    
    def _render_performance_attribution_tab(self):
        """Render performance attribution tab"""
        st.header("Performance Attribution Analysis")
        
        attributor = self._get_attributor()
        if attributor is None:
            st.warning("Please configure portfolio in sidebar")
            return
        
        # Calculate attribution
        attribution = attributor.brinson_attribution()
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Portfolio Return", f"{attribution.total_return:.2%}")
        with col2:
            st.metric("Benchmark Return", f"{attribution.benchmark_return:.2%}")
        with col3:
            st.metric(
                "Active Return",
                f"{attribution.active_return:.2%}",
                delta=f"{attribution.active_return:.2%}"
            )
        with col4:
            ir = attributor.calculate_information_ratio()
            st.metric("Information Ratio", f"{ir:.2f}")
        
        st.markdown("---")
        
        # Brinson attribution breakdown
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Attribution Effects")
            fig = self._create_attribution_effects_chart(attribution)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Active Return Decomposition")
            fig = self._create_decomposition_pie(attribution)
            st.plotly_chart(fig, use_container_width=True)
        
        # Asset contribution waterfall
        st.subheader("Asset Contribution to Portfolio Return")
        fig = self._create_waterfall_chart(attribution)
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed breakdown table
        st.subheader("Detailed Attribution")
        attribution_df = self._create_attribution_table(
            attribution,
            attributor.portfolio_weights,
            attributor.benchmark_weights
        )
        st.dataframe(attribution_df, use_container_width=True)
        
        # Factor attribution
        st.subheader("Factor Attribution")
        factor_contrib = attributor.factor_attribution()
        
        fig = go.Figure()
        factors = list(factor_contrib.keys())
        values = list(factor_contrib.values())
        
        fig.add_trace(go.Bar(
            x=factors,
            y=values,
            marker_color=['green' if v > 0 else 'red' for v in values],
            text=[f"{v:.4f}" for v in values],
            textposition='outside'
        ))
        
        fig.update_layout(
            xaxis_title="Factor",
            yaxis_title="Contribution",
            showlegend=False,
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_risk_attribution_tab(self):
        """Render risk attribution tab"""
        st.header("Risk Attribution Analysis")
        
        attributor = self._get_attributor()
        if attributor is None:
            st.warning("Please configure portfolio in sidebar")
            return
        
        risk_attr = attributor.risk_attribution()
        
        # Key metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Portfolio Risk (ann.)",
                f"{risk_attr.portfolio_risk:.2%}"
            )
        with col2:
            st.metric(
                "Diversification Ratio",
                f"{risk_attr.diversification_ratio:.2f}"
            )
        with col3:
            sharpe = attributor.calculate_sharpe_ratio()
            st.metric("Sharpe Ratio", f"{sharpe:.2f}")
        
        st.markdown("---")
        
        # Risk contribution charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Risk Contribution by Asset")
            fig = self._create_risk_contribution_chart(risk_attr)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Component VaR")
            fig = self._create_component_var_chart(risk_attr)
            st.plotly_chart(fig, use_container_width=True)
        
        # Marginal risk contributions
        st.subheader("Marginal Risk Contributions")
        fig = self._create_marginal_risk_chart(risk_attr)
        st.plotly_chart(fig, use_container_width=True)
        
        # Risk attribution table
        st.subheader("Detailed Risk Attribution")
        risk_df = pd.DataFrame({
            'Asset': list(risk_attr.asset_risk_contributions.keys()),
            'Risk Contribution': list(risk_attr.asset_risk_contributions.values()),
            'Marginal Risk': list(risk_attr.marginal_risk_contributions.values()),
            'Component VaR': list(risk_attr.component_var.values())
        })
        st.dataframe(risk_df, use_container_width=True)
    
    def _render_rolling_analysis_tab(self):
        """Render rolling attribution analysis tab"""
        st.header("Rolling Attribution Analysis")
        
        attributor = self._get_attributor()
        if attributor is None:
            st.warning("Please configure portfolio in sidebar")
            return
        
        # Parameters
        col1, col2 = st.columns(2)
        with col1:
            window = st.slider(
                "Rolling Window (days)",
                30, 365, 252, 30
            )
        with col2:
            step = st.slider(
                "Step Size (days)",
                1, 60, 21, 1
            )
        
        if st.button("Calculate Rolling Attribution", type="primary"):
            with st.spinner("Calculating..."):
                rolling_attr = attributor.rolling_attribution(window=window, step=step)
                st.session_state.rolling_attr = rolling_attr
        
        if hasattr(st.session_state, 'rolling_attr'):
            rolling_attr = st.session_state.rolling_attr
            
            # Rolling returns chart
            st.subheader("Rolling Returns")
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=rolling_attr.index,
                y=rolling_attr['Total Return'] * 100,
                name='Portfolio',
                line=dict(color='blue', width=2)
            ))
            
            fig.add_trace(go.Scatter(
                x=rolling_attr.index,
                y=rolling_attr['Benchmark Return'] * 100,
                name='Benchmark',
                line=dict(color='gray', width=2, dash='dash')
            ))
            
            fig.update_layout(
                xaxis_title="Date",
                yaxis_title="Return (%)",
                height=400,
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Rolling attribution effects
            st.subheader("Rolling Attribution Effects")
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=rolling_attr.index,
                y=rolling_attr['Allocation Effect'] * 100,
                name='Allocation',
                fill='tonexty',
                line=dict(width=0.5)
            ))
            
            fig.add_trace(go.Scatter(
                x=rolling_attr.index,
                y=rolling_attr['Selection Effect'] * 100,
                name='Selection',
                fill='tonexty',
                line=dict(width=0.5)
            ))
            
            fig.add_trace(go.Scatter(
                x=rolling_attr.index,
                y=rolling_attr['Active Return'] * 100,
                name='Active Return',
                line=dict(color='black', width=2)
            ))
            
            fig.update_layout(
                xaxis_title="Date",
                yaxis_title="Effect (%)",
                height=500,
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Summary statistics
            st.subheader("Summary Statistics")
            summary_df = rolling_attr.describe()
            st.dataframe(summary_df, use_container_width=True)
    
    def _render_reports_tab(self):
        """Render reports tab"""
        st.header("Performance Reports")
        
        attributor = self._get_attributor()
        if attributor is None:
            st.warning("Please configure portfolio in sidebar")
            return
        
        # Generate text report
        report = attributor.generate_attribution_report()
        
        st.subheader("Attribution Report")
        st.text(report)
        
        # Download button
        st.download_button(
            label="📥 Download Report",
            data=report,
            file_name="attribution_report.txt",
            mime="text/plain"
        )
        
        st.markdown("---")
        
        # Asset contributions over time
        st.subheader("Asset Contributions (Monthly)")
        monthly_contrib = attributor.calculate_asset_contributions(period='monthly')
        
        fig = go.Figure()
        for asset in monthly_contrib.columns:
            fig.add_trace(go.Scatter(
                x=monthly_contrib.index,
                y=monthly_contrib[asset] * 100,
                name=asset,
                mode='lines'
            ))
        
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Contribution (%)",
            height=500,
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    def _create_attribution_effects_chart(self, attribution: AttributionResult):
        """Create attribution effects bar chart"""
        fig = go.Figure()
        
        categories = ['Allocation', 'Selection', 'Interaction']
        values = [
            attribution.allocation_effect * 100,
            attribution.selection_effect * 100,
            attribution.interaction_effect * 100
        ]
        
        colors = ['blue', 'green', 'orange']
        
        fig.add_trace(go.Bar(
            x=categories,
            y=values,
            marker_color=colors,
            text=[f"{v:.2f}%" for v in values],
            textposition='outside'
        ))
        
        fig.update_layout(
            yaxis_title="Effect (%)",
            showlegend=False,
            height=400
        )
        
        return fig
    
    def _create_decomposition_pie(self, attribution: AttributionResult):
        """Create active return decomposition pie chart"""
        labels = ['Allocation', 'Selection', 'Interaction']
        values = [
            abs(attribution.allocation_effect),
            abs(attribution.selection_effect),
            abs(attribution.interaction_effect)
        ]
        
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=0.3
        )])
        
        fig.update_layout(height=400)
        
        return fig
    
    def _create_waterfall_chart(self, attribution: AttributionResult):
        """Create waterfall chart for asset contributions"""
        sorted_contrib = sorted(
            attribution.asset_contributions.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        assets = [item[0] for item in sorted_contrib]
        values = [item[1] * 100 for item in sorted_contrib]
        
        # Calculate cumulative
        cumulative = [0]
        for v in values:
            cumulative.append(cumulative[-1] + v)
        
        fig = go.Figure()
        
        # Bars
        for i, (asset, value) in enumerate(zip(assets, values)):
            fig.add_trace(go.Bar(
                x=[asset],
                y=[value],
                name=asset,
                text=f"{value:.2f}%",
                textposition='outside',
                marker_color='green' if value > 0 else 'red'
            ))
        
        # Total line
        fig.add_trace(go.Scatter(
            x=assets + ['Total'],
            y=cumulative,
            mode='lines+markers',
            name='Cumulative',
            line=dict(color='blue', width=2)
        ))
        
        fig.update_layout(
            xaxis_title="Asset",
            yaxis_title="Contribution (%)",
            showlegend=False,
            height=500
        )
        
        return fig
    
    def _create_attribution_table(
        self,
        attribution: AttributionResult,
        portfolio_weights: Dict[str, float],
        benchmark_weights: Dict[str, float]
    ) -> pd.DataFrame:
        """Create detailed attribution table"""
        data = []
        
        for asset in attribution.asset_contributions.keys():
            data.append({
                'Asset': asset,
                'Portfolio Weight': f"{portfolio_weights[asset]:.2%}",
                'Benchmark Weight': f"{benchmark_weights[asset]:.2%}",
                'Contribution': f"{attribution.asset_contributions[asset]:.4f}"
            })
        
        return pd.DataFrame(data)
    
    def _create_risk_contribution_chart(self, risk_attr: RiskAttribution):
        """Create risk contribution bar chart"""
        fig = go.Figure()
        
        assets = list(risk_attr.asset_risk_contributions.keys())
        values = [v * 100 for v in risk_attr.asset_risk_contributions.values()]
        
        fig.add_trace(go.Bar(
            x=assets,
            y=values,
            marker_color='steelblue',
            text=[f"{v:.2f}%" for v in values],
            textposition='outside'
        ))
        
        fig.update_layout(
            xaxis_title="Asset",
            yaxis_title="Risk Contribution (%)",
            showlegend=False,
            height=400
        )
        
        return fig
    
    def _create_component_var_chart(self, risk_attr: RiskAttribution):
        """Create component VaR chart"""
        fig = go.Figure(data=[go.Pie(
            labels=list(risk_attr.component_var.keys()),
            values=[abs(v) * 100 for v in risk_attr.component_var.values()],
            hole=0.3
        )])
        
        fig.update_layout(height=400)
        
        return fig
    
    def _create_marginal_risk_chart(self, risk_attr: RiskAttribution):
        """Create marginal risk contributions chart"""
        fig = go.Figure()
        
        assets = list(risk_attr.marginal_risk_contributions.keys())
        values = [v * 100 for v in risk_attr.marginal_risk_contributions.values()]
        
        fig.add_trace(go.Bar(
            x=assets,
            y=values,
            marker_color='coral',
            text=[f"{v:.2f}%" for v in values],
            textposition='outside'
        ))
        
        fig.update_layout(
            xaxis_title="Asset",
            yaxis_title="Marginal Risk (%)",
            showlegend=False,
            height=400
        )
        
        return fig


def main():
    """Main function to run dashboard"""
    dashboard = AttributionDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()
