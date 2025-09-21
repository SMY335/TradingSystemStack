from __future__ import annotations
import pandas as pd
import vectorbt as vbt

def ema_cross_signals(close: pd.Series, fast=20, slow=50):
    fast_ema = vbt.MA.run(close, window=fast, ewm=True).ma
    slow_ema = vbt.MA.run(close, window=slow, ewm=True).ma
    entries = (fast_ema > slow_ema) & (fast_ema.shift(1) <= slow_ema.shift(1))
    exits   = (fast_ema < slow_ema) & (fast_ema.shift(1) >= slow_ema.shift(1))
    return entries, exits

def run_backtest(df: pd.DataFrame, fast=20, slow=50):
    entries, exits = ema_cross_signals(df["Close"], fast, slow)
    pf = vbt.Portfolio.from_signals(df["Close"], entries, exits, fees=0.0002, slippage=0.0001)
    kpis = {
        "winrate_pct": float(pf.trades.win_rate * 100),
        "profit_factor": float(pf.trades.profit_factor),
        "max_drawdown_pct": float(abs(pf.drawdown_series().min()) * 100),
        "trades": int(pf.trades.count())
    }
    return pf, kpis
