#!/bin/bash
set -e

echo "🚀 Setting up TradingSystemStack..."

# Update pip
pip install --upgrade pip

# Install system dependencies for TA-Lib
echo "📦 Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y wget build-essential

# Install TA-Lib C library
echo "📊 Installing TA-Lib..."
cd /tmp
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make
sudo make install
cd /workspaces/TradingSystemStack

# Install Python dependencies
echo "🐍 Installing Python packages..."
pip install -r requirements_frameworks.txt 2>/dev/null || echo "Note: Some packages may have warnings"

# Install additional dependencies from pyproject.toml
pip install pandas numpy numba vectorbt quantstats empyrical-reloaded
pip install prefect pydantic pyyaml ccxt ib-async
pip install freqtrade riskfolio-lib streamlit plotly scikit-learn scipy

# Make shell scripts executable
chmod +x *.sh

# Create necessary directories
mkdir -p logs data

echo "✅ Setup complete!"
echo ""
echo "🎯 Quick Start Commands:"
echo "  ./run_dashboard.sh           - Backtesting Dashboard"
echo "  ./run_live_dashboard.sh      - Live Trading Dashboard"
echo "  ./run_portfolio_dashboard.sh - Portfolio Management"
echo ""
echo "📚 Check README.md for more information"
