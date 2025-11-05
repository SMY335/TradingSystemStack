"""
Market Maker Strategy Adapter

Adapts the Market Maker strategy for use with different frameworks.
"""

from src.adapters.base_strategy_adapter import BaseStrategyAdapter, StrategyConfig
from src.market_making.simple_mm import SimpleMarketMaker


class MarketMakerAdapter(BaseStrategyAdapter):
    """Adapter for Market Maker strategy"""
    
    def validate_parameters(self):
        """Validate Market Maker specific parameters"""
        required = ['spread_bps', 'max_inventory', 'order_size']
        for param in required:
            if param not in self.config.parameters:
                raise ValueError(f"Missing required parameter: {param}")
        
        # Validate parameter ranges
        if self.config.parameters['spread_bps'] < 1:
            raise ValueError("spread_bps must be >= 1")
        
        if self.config.parameters['max_inventory'] < 1:
            raise ValueError("max_inventory must be >= 1")
        
        if self.config.parameters['order_size'] <= 0:
            raise ValueError("order_size must be > 0")
    
    def to_nautilus(self):
        """Convert to Nautilus Trader strategy"""
        # TODO: Implement Nautilus version when needed
        raise NotImplementedError("Market Maker Nautilus adapter to be implemented")
    
    def to_backtrader(self):
        """Return Backtrader strategy class"""
        return SimpleMarketMaker
    
    def get_parameter_space(self):
        """Return parameter space for optimization"""
        return {
            'spread_bps': [10, 15, 20, 25, 30],
            'max_inventory': [5, 10, 15, 20],
            'order_size': [0.5, 1.0, 1.5, 2.0],
            'skew_factor': [0.0005, 0.001, 0.0015, 0.002],
            'update_freq': [1, 2, 5],
        }
    
    def get_default_config(self) -> StrategyConfig:
        """Return default configuration for Market Maker"""
        return StrategyConfig(
            name="MarketMaker_Default",
            strategy_type="market_maker",
            parameters={
                'spread_bps': 20,
                'max_inventory': 10,
                'order_size': 1.0,
                'skew_factor': 0.001,
                'update_freq': 1,
            },
            risk_management={
                'max_inventory': 10,
                'max_spread': 100,  # bps
            }
        )
