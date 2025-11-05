"""
Pairs Trading Strategy Adapter

Adapts the Pairs Trading strategy for use with different frameworks.
"""

from src.adapters.base_strategy_adapter import BaseStrategyAdapter, StrategyConfig
from src.quant_strategies.pairs_trading import PairsTradingStrategy


class PairsAdapter(BaseStrategyAdapter):
    """Adapter for Pairs Trading strategy"""
    
    def validate_parameters(self):
        """Validate Pairs Trading specific parameters"""
        required = ['lookback', 'entry_zscore', 'exit_zscore']
        for param in required:
            if param not in self.config.parameters:
                raise ValueError(f"Missing required parameter: {param}")
        
        # Validate parameter ranges
        if self.config.parameters['lookback'] < 20:
            raise ValueError("lookback must be >= 20")
        
        if self.config.parameters['entry_zscore'] <= self.config.parameters['exit_zscore']:
            raise ValueError("entry_zscore must be > exit_zscore")
        
        if self.config.parameters.get('stop_loss_zscore', 999) <= self.config.parameters['entry_zscore']:
            raise ValueError("stop_loss_zscore must be > entry_zscore")
    
    def to_nautilus(self):
        """Convert to Nautilus Trader strategy"""
        # TODO: Implement Nautilus version when needed
        raise NotImplementedError("Pairs Trading Nautilus adapter to be implemented")
    
    def to_backtrader(self):
        """Return Backtrader strategy class"""
        return PairsTradingStrategy
    
    def get_parameter_space(self):
        """Return parameter space for optimization"""
        return {
            'lookback': [40, 60, 80, 100],
            'entry_zscore': [1.5, 2.0, 2.5, 3.0],
            'exit_zscore': [0.25, 0.5, 0.75, 1.0],
            'stop_loss_zscore': [3.0, 3.5, 4.0],
            'recalc_period': [10, 20, 30],
        }
    
    def get_default_config(self) -> StrategyConfig:
        """Return default configuration for Pairs Trading"""
        return StrategyConfig(
            name="PairsTrading_Default",
            strategy_type="pairs_trading",
            parameters={
                'lookback': 60,
                'entry_zscore': 2.0,
                'exit_zscore': 0.5,
                'stop_loss_zscore': 3.0,
                'position_size': 1.0,
                'recalc_period': 20,
            },
            risk_management={
                'max_position_size': 1.0,
                'max_spread_deviation': 3.0,
            }
        )
