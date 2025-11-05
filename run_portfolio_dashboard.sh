#!/bin/bash

# Portfolio Management Dashboard Launcher

echo "🚀 Launching Portfolio Management Dashboard..."
echo ""
echo "📊 Dashboard will open at: http://localhost:8503"
echo ""
echo "Features:"
echo "  ✓ Portfolio Overview & Analytics"
echo "  ✓ Multi-Strategy Optimization"
echo "  ✓ Automatic Rebalancing"
echo "  ✓ Risk Metrics & Performance Attribution"
echo ""

cd "$(dirname "$0")"
streamlit run src/dashboard/portfolio_dashboard.py --server.port 8503
