"""
Fundamental Data Module.

This module provides access to fundamental financial data including
financial statements, ratios, and metrics for stocks.

Examples:
    >>> from src.fundamentals import get_financial_ratios, FinancialAnalyzer
    >>>
    >>> # Get key ratios
    >>> ratios = get_financial_ratios('AAPL')
    >>> print(f"P/E Ratio: {ratios['pe_ratio']}")
    >>>
    >>> # Comprehensive analysis
    >>> analyzer = FinancialAnalyzer()
    >>> analysis = analyzer.analyze('AAPL')
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime


@dataclass
class FinancialRatios:
    """Financial ratios for a company."""
    symbol: str
    timestamp: datetime

    # Valuation ratios
    pe_ratio: Optional[float] = None
    pb_ratio: Optional[float] = None
    ps_ratio: Optional[float] = None
    peg_ratio: Optional[float] = None

    # Profitability ratios
    roe: Optional[float] = None  # Return on Equity
    roa: Optional[float] = None  # Return on Assets
    roic: Optional[float] = None  # Return on Invested Capital
    gross_margin: Optional[float] = None
    operating_margin: Optional[float] = None
    net_margin: Optional[float] = None

    # Liquidity ratios
    current_ratio: Optional[float] = None
    quick_ratio: Optional[float] = None

    # Leverage ratios
    debt_to_equity: Optional[float] = None
    debt_to_assets: Optional[float] = None

    # Efficiency ratios
    asset_turnover: Optional[float] = None
    inventory_turnover: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp,
            'pe_ratio': self.pe_ratio,
            'pb_ratio': self.pb_ratio,
            'ps_ratio': self.ps_ratio,
            'peg_ratio': self.peg_ratio,
            'roe': self.roe,
            'roa': self.roa,
            'roic': self.roic,
            'gross_margin': self.gross_margin,
            'operating_margin': self.operating_margin,
            'net_margin': self.net_margin,
            'current_ratio': self.current_ratio,
            'quick_ratio': self.quick_ratio,
            'debt_to_equity': self.debt_to_equity,
            'debt_to_assets': self.debt_to_assets,
            'asset_turnover': self.asset_turnover,
            'inventory_turnover': self.inventory_turnover,
        }


def get_financial_ratios(
    symbol: str,
    use_mock: bool = True
) -> FinancialRatios:
    """
    Get financial ratios for a stock.

    Args:
        symbol: Stock symbol
        use_mock: Use mock data (True) or real API (False)

    Returns:
        FinancialRatios object

    Examples:
        >>> ratios = get_financial_ratios('AAPL')
        >>> print(f"P/E: {ratios.pe_ratio:.2f}")
    """
    if use_mock:
        # Generate realistic mock data
        np.random.seed(hash(symbol) % 2**32)

        return FinancialRatios(
            symbol=symbol,
            timestamp=datetime.now(),
            pe_ratio=15 + np.random.randn() * 5,
            pb_ratio=3 + np.random.randn() * 1,
            ps_ratio=2 + np.random.randn() * 0.5,
            peg_ratio=1.5 + np.random.randn() * 0.5,
            roe=0.15 + np.random.randn() * 0.05,
            roa=0.10 + np.random.randn() * 0.03,
            roic=0.12 + np.random.randn() * 0.04,
            gross_margin=0.40 + np.random.randn() * 0.1,
            operating_margin=0.20 + np.random.randn() * 0.05,
            net_margin=0.15 + np.random.randn() * 0.05,
            current_ratio=1.5 + np.random.randn() * 0.3,
            quick_ratio=1.2 + np.random.randn() * 0.3,
            debt_to_equity=0.5 + np.random.randn() * 0.2,
            debt_to_assets=0.3 + np.random.randn() * 0.1,
            asset_turnover=0.8 + np.random.randn() * 0.2,
            inventory_turnover=6 + np.random.randn() * 2
        )
    else:
        # Real implementation would use yfinance, Financial Modeling Prep API, etc.
        raise NotImplementedError("Real API integration not yet implemented")


def calculate_intrinsic_value_dcf(
    free_cash_flow: float,
    growth_rate: float,
    discount_rate: float,
    terminal_growth: float = 0.03,
    years: int = 5
) -> float:
    """
    Calculate intrinsic value using DCF model.

    Args:
        free_cash_flow: Current free cash flow
        growth_rate: Expected growth rate (as decimal)
        discount_rate: Discount rate / WACC (as decimal)
        terminal_growth: Terminal growth rate (as decimal)
        years: Number of years to project

    Returns:
        Estimated intrinsic value

    Examples:
        >>> value = calculate_intrinsic_value_dcf(
        ...     free_cash_flow=1000,
        ...     growth_rate=0.10,
        ...     discount_rate=0.08
        ... )
    """
    present_value = 0.0

    # Project cash flows
    for year in range(1, years + 1):
        fcf = free_cash_flow * ((1 + growth_rate) ** year)
        pv = fcf / ((1 + discount_rate) ** year)
        present_value += pv

    # Terminal value
    terminal_fcf = free_cash_flow * ((1 + growth_rate) ** years) * (1 + terminal_growth)
    terminal_value = terminal_fcf / (discount_rate - terminal_growth)
    terminal_pv = terminal_value / ((1 + discount_rate) ** years)

    intrinsic_value = present_value + terminal_pv

    return intrinsic_value


def screen_by_fundamentals(
    symbols: List[str],
    criteria: Dict[str, tuple],
    use_mock: bool = True
) -> List[str]:
    """
    Screen stocks by fundamental criteria.

    Args:
        symbols: List of symbols to screen
        criteria: Dict of {metric: (min, max)} ranges
        use_mock: Use mock data

    Returns:
        List of symbols that pass all criteria

    Examples:
        >>> passed = screen_by_fundamentals(
        ...     ['AAPL', 'MSFT', 'GOOGL'],
        ...     {'pe_ratio': (0, 25), 'roe': (0.15, 1.0)}
        ... )
    """
    passed = []

    for symbol in symbols:
        ratios = get_financial_ratios(symbol, use_mock=use_mock)
        ratios_dict = ratios.to_dict()

        meets_criteria = True

        for metric, (min_val, max_val) in criteria.items():
            value = ratios_dict.get(metric)

            if value is None:
                meets_criteria = False
                break

            # Check min value if specified
            if min_val is not None and value < min_val:
                meets_criteria = False
                break

            # Check max value if specified
            if max_val is not None and value > max_val:
                meets_criteria = False
                break

        if meets_criteria:
            passed.append(symbol)

    return passed


class FinancialAnalyzer:
    """
    Comprehensive fundamental analysis.

    Examples:
        >>> analyzer = FinancialAnalyzer()
        >>> analysis = analyzer.analyze('AAPL')
        >>> print(f"Score: {analysis['overall_score']}")
    """

    def __init__(self, use_mock: bool = True):
        self.use_mock = use_mock

    def analyze(self, symbol: str) -> Dict[str, Any]:
        """
        Perform comprehensive fundamental analysis.

        Args:
            symbol: Stock symbol

        Returns:
            Dictionary with analysis results

        Examples:
            >>> analysis = analyzer.analyze('AAPL')
        """
        ratios = get_financial_ratios(symbol, use_mock=self.use_mock)

        # Score different aspects
        valuation_score = self._score_valuation(ratios)
        profitability_score = self._score_profitability(ratios)
        liquidity_score = self._score_liquidity(ratios)
        leverage_score = self._score_leverage(ratios)

        # Overall score (0-100)
        overall_score = (
            valuation_score * 0.25 +
            profitability_score * 0.35 +
            liquidity_score * 0.20 +
            leverage_score * 0.20
        )

        # Determine rating
        if overall_score >= 80:
            rating = 'Strong Buy'
        elif overall_score >= 65:
            rating = 'Buy'
        elif overall_score >= 50:
            rating = 'Hold'
        elif overall_score >= 35:
            rating = 'Sell'
        else:
            rating = 'Strong Sell'

        return {
            'symbol': symbol,
            'timestamp': ratios.timestamp,
            'valuation_score': valuation_score,
            'profitability_score': profitability_score,
            'liquidity_score': liquidity_score,
            'leverage_score': leverage_score,
            'overall_score': overall_score,
            'rating': rating,
            'ratios': ratios.to_dict()
        }

    def _score_valuation(self, ratios: FinancialRatios) -> float:
        """Score valuation (0-100)."""
        score = 50.0  # Start neutral

        # P/E ratio (lower is better, 15-20 is reasonable)
        if ratios.pe_ratio is not None:
            if ratios.pe_ratio < 15:
                score += 20
            elif ratios.pe_ratio < 25:
                score += 10
            elif ratios.pe_ratio > 40:
                score -= 20

        # PEG ratio (closer to 1 is better)
        if ratios.peg_ratio is not None:
            if 0.8 <= ratios.peg_ratio <= 1.2:
                score += 15
            elif ratios.peg_ratio > 2:
                score -= 15

        return np.clip(score, 0, 100)

    def _score_profitability(self, ratios: FinancialRatios) -> float:
        """Score profitability (0-100)."""
        score = 50.0

        # ROE (higher is better)
        if ratios.roe is not None:
            if ratios.roe > 0.20:
                score += 25
            elif ratios.roe > 0.15:
                score += 15
            elif ratios.roe < 0.10:
                score -= 15

        # Net margin
        if ratios.net_margin is not None:
            if ratios.net_margin > 0.20:
                score += 25
            elif ratios.net_margin > 0.10:
                score += 15
            elif ratios.net_margin < 0.05:
                score -= 15

        return np.clip(score, 0, 100)

    def _score_liquidity(self, ratios: FinancialRatios) -> float:
        """Score liquidity (0-100)."""
        score = 50.0

        # Current ratio (>1.5 is good)
        if ratios.current_ratio is not None:
            if ratios.current_ratio > 2:
                score += 25
            elif ratios.current_ratio > 1.5:
                score += 15
            elif ratios.current_ratio < 1:
                score -= 25

        return np.clip(score, 0, 100)

    def _score_leverage(self, ratios: FinancialRatios) -> float:
        """Score leverage (0-100)."""
        score = 50.0

        # Debt to equity (lower is better)
        if ratios.debt_to_equity is not None:
            if ratios.debt_to_equity < 0.3:
                score += 25
            elif ratios.debt_to_equity < 0.6:
                score += 15
            elif ratios.debt_to_equity > 1.5:
                score -= 25

        return np.clip(score, 0, 100)


__all__ = [
    'FinancialRatios',
    'get_financial_ratios',
    'calculate_intrinsic_value_dcf',
    'screen_by_fundamentals',
    'FinancialAnalyzer',
]

__version__ = '2.0.0'

# Note: Real implementation would integrate with:
# - yfinance for basic fundamentals
# - financialtoolkit for comprehensive analysis
# - Financial Modeling Prep API
# - Alpha Vantage Fundamentals API
