#!/bin/bash

# Launch Risk Management Dashboard
# This script starts the Streamlit dashboard for portfolio risk analysis

echo "=========================================="
echo "Risk Management Dashboard"
echo "=========================================="
echo ""
echo "Starting Streamlit dashboard..."
echo "Dashboard will open in your browser automatically."
echo ""
echo "Features:"
echo "  - VaR & CVaR analysis"
echo "  - Monte Carlo simulation"
echo "  - Stress testing"
echo "  - Scenario analysis"
echo ""
echo "Press Ctrl+C to stop the dashboard"
echo ""

streamlit run src/dashboard/risk_dashboard.py
