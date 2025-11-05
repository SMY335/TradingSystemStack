"""
ICT Strategy Adapter

Adapts the Inner Circle Trader strategy for use with different frameworks.
"""

from src.adapters.base_strategy_adapter import BaseStrategyAdapter, StrategyConfig
from src.ict_strategies.ict_strategy import ICTStrategy


class ICTAdapter(BaseStrategyAdapter):
    """Adapter for ICT trading strategy"""
    
    def validate_parameters(self):
        """Validate ICT-specific parameters"""
        required = ['order_block_lookback', 'liquidity_lookback', 'risk_reward_ratio']
        for param in required:
            if param not in self.config.parameters:
                raise ValueError(f"Missing required parameter: {param}")
        
        # Validate parameter ranges
        if self.config.parameters['order_block_lookback'] < 10:
            raise ValueError("order_block_lookback must be >= 10")
        
        if self.config.parameters['liquidity_lookback'] < 20:
            raise ValueError("liquidity_lookback must be >= 20")
        
        if self.config.parameters['risk_reward_ratio'] < 1.0:
            raise ValueError("risk_reward_ratio must be >= 1.0")
    
    def to_nautilus(self):
        """Convert to Nautilus Trader strategy"""
        # TODO: Implement Nautilus version when needed
        raise NotImplementedError("ICT Nautilus adapter to be implemented")
    
    def to_backtrader(self):
        """Return Backtrader strategy class"""
        return ICTStrategy
    
    def get_parameter_space(self):
        """Return parameter space for optimization"""
        return {
            'order_block_lookback': [10, 15, 20, 25, 30],
            'fvg_min_gap_pct': [0.0005, 0.001, 0.0015, 0.002],
            'liquidity_lookback': [30, 40, 50, 60, 70],
            'risk_reward_ratio': [1.5, 2.0, 2.5, 3.0],
            'position_size_pct': [0.01, 0.02, 0.03],
        }
    
    def get_default_config(self) -> StrategyConfig:
        """Return default configuration for ICT strategy"""
        return StrategyConfig(
            name="ICT_Default",
            strategy_type="ict",
            parameters={
                'order_block_lookback': 20,
                'fvg_min_gap_pct': 0.001,
                'liquidity_lookback': 50,
                'risk_reward_ratio': 2.0,
                'position_size_pct': 0.02,
                'max_positions': 1,
            },
            risk_management={
                'max_position_size': 0.02,
                'stop_loss_pct': 0.005,
                'take_profit_pct': 0.01,
            }
        )
