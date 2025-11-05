#!/bin/bash

# Run Risk Management Dashboard
# Phase 5 - Session 19: Advanced Risk Management

echo "🚀 Launching Risk Management Dashboard..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Features:"
echo "  ⚠️  VaR & CVaR Analysis"
echo "  🎲 Monte Carlo Simulation"
echo "  💥 Stress Testing"
echo "  📊 Risk Metrics & Attribution"
echo ""
echo "Opening at: http://localhost:8504"
echo "Press Ctrl+C to stop"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

streamlit run src/dashboard/risk_dashboard.py --server.port 8504
