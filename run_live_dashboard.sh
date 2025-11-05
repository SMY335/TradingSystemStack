#!/bin/bash
# Launch Live Paper Trading Dashboard

echo "🤖 Launching Live Paper Trading Dashboard..."
echo "📊 Dashboard will open in your browser"
echo "🔄 The dashboard auto-refreshes every 5 seconds"
echo ""
echo "Press Ctrl+C to stop"
echo ""

streamlit run src/dashboard/live_dashboard.py --server.port 8502 --server.address 0.0.0.0
