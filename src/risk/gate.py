from __future__ import annotations
import operator
from typing import Dict

OPS = {">": operator.gt, "<": operator.lt, ">=": operator.ge, "<=": operator.le}

def _parse(rule: str) -> tuple[str, float]:
    if rule[:2] in (">=", "<="): return rule[:2], float(rule[2:])
    return rule[0], float(rule[1:])

def enforce_policy(kpis: Dict[str, float], policy: Dict) -> None:
    th = policy["kpi_thresholds"]
    checks = {
        "winrate_pct": kpis["winrate_pct"],
        "profit_factor": kpis["profit_factor"],
        "max_drawdown_pct": kpis["max_drawdown_pct"],
        "trades_min": kpis["trades"]
    }
    for key, rule in th.items():
        op, val = _parse(rule)
        if not OPS[op](checks[key], val):
            raise AssertionError(f"Risk gate FAIL: {key} {rule} (got {checks[key]})")
