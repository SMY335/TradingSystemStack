#!/bin/bash
# Launch the Trading Bot Dashboard

echo "🚀 Launching Trading Bot Dashboard..."
echo "📊 Dashboard will open in your browser"
echo ""

streamlit run src/dashboard/app.py --server.port 8501 --server.address 0.0.0.0
