import pandas as pd
from src.research.vectorbt_strategies import run_backtest

def test_smoke_backtest():
    idx = pd.date_range("2024-01-01", periods=120, freq="H")
    close = pd.Series(100 + (pd.Series(range(120))*0.03).values, index=idx, name="Close")
    df = close.to_frame()
    _, kpis = run_backtest(df)
    assert kpis["trades"] >= 1
