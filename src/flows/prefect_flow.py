from __future__ import annotations
from prefect import flow, task
import yaml, pandas as pd
from pathlib import Path
from src.research.vectorbt_strategies import run_backtest
from src.risk.gate import enforce_policy

@task
def ingest(path="data/fx/demo_eurusd_h1.csv") -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"]).set_index("Date")
    return df

@task
def backtest(df: pd.DataFrame):
    return run_backtest(df)

@task
def policy_check(kpis, policy_path="infra/policy.yaml"):
    policy = yaml.safe_load(Path(policy_path).read_text(encoding="utf-8"))
    enforce_policy(kpis, policy)

@flow
def e2e():
    df = ingest()
    pf, kpis = backtest(df)
    policy_check(kpis)
    pf.stats().to_csv("reports_stats.csv")

if __name__ == "__main__":
    e2e()
