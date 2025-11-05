#!/bin/bash

# Launch Performance Attribution Dashboard
# This script starts the Streamlit dashboard for performance attribution analysis

echo "=========================================="
echo "Performance Attribution Dashboard"
echo "=========================================="
echo ""
echo "Starting Streamlit dashboard..."
echo "Dashboard will open in your browser automatically."
echo ""
echo "Features:"
echo "  - Brinson attribution"
echo "  - Factor attribution"
echo "  - Risk attribution"
echo "  - Rolling analysis"
echo ""
echo "Press Ctrl+C to stop the dashboard"
echo ""

streamlit run src/dashboard/attribution_dashboard.py
