#!/bin/bash

# Run Performance Attribution Dashboard
# Phase 5 - Session 20: Performance Attribution

echo "🚀 Launching Performance Attribution Dashboard..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Features:"
echo "  📊 Brinson Attribution"
echo "  💼 Asset Contribution Analysis"
echo "  📈 Rolling Attribution"
echo "  🎯 Factor Attribution"
echo ""
echo "Opening at: http://localhost:8505"
echo "Press Ctrl+C to stop"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

streamlit run src/dashboard/attribution_dashboard.py --server.port 8505
