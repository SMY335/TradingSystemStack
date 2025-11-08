"""
Scanner execution engine.

This module executes scan definitions across symbols,
optionally in parallel.
"""

import pandas as pd
from typing import List, Dict, Any
from datetime import datetime
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from .dsl import ScanDefinition, ScanResult, ScanResults, Condition
from .operators import (
    compare,
    crosses_above,
    crosses_below,
    logical_and,
    logical_or,
    check_pattern,
    get_value
)
from src.data import get_ohlcv
from src.indicators import run_indicator


class ScanEngine:
    """
    Engine for executing market scans.

    Examples:
        >>> engine = ScanEngine()
        >>> scan = load_scan_from_json('scans/rsi_oversold.json')
        >>> results = engine.execute(scan)
        >>> print(f"Found {len(results.matched)} matches")
    """

    def __init__(self, max_workers: int = 4, verbose: bool = False):
        """
        Initialize scan engine.

        Args:
            max_workers: Maximum parallel workers
            verbose: Print progress messages
        """
        self.max_workers = max_workers
        self.verbose = verbose

    def execute(self, scan: ScanDefinition) -> ScanResults:
        """
        Execute a scan across all symbols.

        Args:
            scan: Scan definition

        Returns:
            ScanResults object

        Examples:
            >>> results = engine.execute(scan)
        """
        start_time = time.time()

        if self.verbose:
            print(f"Executing scan: {scan.name}")
            print(f"Universe: {len(scan.universe)} symbols")

        matched = []

        # Execute scan for each symbol (optionally in parallel)
        if self.max_workers > 1:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(self._scan_symbol, symbol, scan): symbol
                    for symbol in scan.universe
                }

                for future in as_completed(futures):
                    result = future.result()
                    if result and result.matched:
                        matched.append(result)
        else:
            for symbol in scan.universe:
                result = self._scan_symbol(symbol, scan)
                if result and result.matched:
                    matched.append(result)

        # Sort results if requested
        if scan.sort_by and matched:
            matched.sort(
                key=lambda r: r.values.get(scan.sort_by, 0),
                reverse=not scan.sort_ascending
            )

        # Limit results if requested
        if scan.max_results:
            matched = matched[:scan.max_results]

        execution_time = time.time() - start_time

        if self.verbose:
            print(f"Scan complete: {len(matched)} matches in {execution_time:.2f}s")

        return ScanResults(
            scan_name=scan.name,
            total_scanned=len(scan.universe),
            matched=matched,
            execution_time=execution_time,
            timestamp=datetime.now()
        )

    def _scan_symbol(self, symbol: str, scan: ScanDefinition) -> ScanResult:
        """Scan a single symbol."""
        try:
            # Load data
            df = get_ohlcv(
                symbol=symbol,
                interval=scan.timeframe
            )

            if len(df) < scan.lookback:
                return ScanResult(
                    symbol=symbol,
                    matched=False,
                    timestamp=datetime.now(),
                    values={}
                )

            # Take only lookback bars
            df = df.tail(scan.lookback).copy()

            # Calculate indicators
            for indicator in scan.indicators:
                result_df = run_indicator(
                    indicator_name=indicator.name,
                    df=df,
                    params=indicator.params
                )

                # Merge indicator columns
                for col in result_df.columns:
                    if col not in df.columns:
                        df[col] = result_df[col]

                # Add alias if specified
                if indicator.alias:
                    # Find the main indicator column
                    indicator_col = indicator.name.lower()
                    if indicator_col in df.columns:
                        df[indicator.alias] = df[indicator_col]

            # Evaluate conditions
            condition_result = self._evaluate_condition(df, scan.conditions)

            # Check if condition is True at latest bar
            matched = bool(condition_result.iloc[-1]) if len(condition_result) > 0 else False

            # Gather relevant values
            values = {
                'close': float(df['close'].iloc[-1]),
                'volume': float(df['volume'].iloc[-1]) if 'volume' in df.columns else 0,
            }

            # Add indicator values
            for indicator in scan.indicators:
                alias = indicator.alias or indicator.name.lower()
                if alias in df.columns:
                    values[alias] = float(df[alias].iloc[-1])

            return ScanResult(
                symbol=symbol,
                matched=matched,
                timestamp=df.index[-1],
                values=values
            )

        except Exception as e:
            if self.verbose:
                print(f"Error scanning {symbol}: {e}")

            return ScanResult(
                symbol=symbol,
                matched=False,
                timestamp=datetime.now(),
                values={}
            )

    def _evaluate_condition(self, df: pd.DataFrame, condition: Condition) -> pd.Series:
        """
        Recursively evaluate a condition tree.

        Args:
            df: DataFrame with data
            condition: Condition to evaluate

        Returns:
            Boolean Series
        """
        cond_type = condition.type

        if cond_type == 'comparison':
            # Get left operand
            left = get_value(df, condition.left)

            # Get right operand
            if isinstance(condition.right, str):
                # It's a column reference
                right = get_value(df, condition.right)
            else:
                # It's a literal value
                right = condition.right

            return compare(left, condition.operator, right)

        elif cond_type == 'cross':
            series1 = get_value(df, condition.series1)
            series2 = get_value(df, condition.series2)

            if condition.direction == 'above':
                return crosses_above(series1, series2, condition.lookback)
            else:
                return crosses_below(series1, series2, condition.lookback)

        elif cond_type == 'pattern':
            return check_pattern(df, condition.pattern_type)

        elif cond_type == 'and':
            # Evaluate all sub-conditions
            results = [
                self._evaluate_condition(df, sub_cond)
                for sub_cond in condition.conditions
            ]
            return logical_and(results)

        elif cond_type == 'or':
            # Evaluate all sub-conditions
            results = [
                self._evaluate_condition(df, sub_cond)
                for sub_cond in condition.conditions
            ]
            return logical_or(results)

        else:
            raise ValueError(f"Unknown condition type: {cond_type}")


def run_scan(
    scan_definition: ScanDefinition,
    max_workers: int = 4,
    verbose: bool = False
) -> ScanResults:
    """
    Convenience function to run a scan.

    Args:
        scan_definition: Scan to execute
        max_workers: Number of parallel workers
        verbose: Print progress

    Returns:
        ScanResults

    Examples:
        >>> from src.scanner import load_scan_from_json, run_scan
        >>> scan = load_scan_from_json('scans/golden_cross.json')
        >>> results = run_scan(scan, verbose=True)
    """
    engine = ScanEngine(max_workers=max_workers, verbose=verbose)
    return engine.execute(scan_definition)
