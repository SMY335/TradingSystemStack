"""
Base Strategy Adapter Interface
Provides unified interface for strategies across different frameworks
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum


class StrategyFramework(Enum):
    """Supported trading frameworks"""
    NAUTILUS = "nautilus"
    BACKTRADER = "backtrader"
    VECTORBT = "vectorbt"


@dataclass
class StrategyConfig:
    """Configuration for strategy instantiation"""
    name: str
    parameters: Dict[str, Any]
    timeframe: str
    symbols: list[str]
    capital: float
    framework: StrategyFramework
    risk_params: Optional[Dict[str, float]] = None
    
    def __post_init__(self):
        """Initialize default risk parameters if not provided"""
        if self.risk_params is None:
            self.risk_params = {
                'max_position_size': 0.1,
                'stop_loss_pct': 0.02
            }


class BaseStrategyAdapter(ABC):
    """
    Abstract base class for strategy adapters
    Converts strategy logic between different trading frameworks
    """
    
    def __init__(self, config: StrategyConfig):
        """
        Initialize strategy adapter
        
        Args:
            config: Strategy configuration
        """
        self.config = config
        self.validate_parameters()
    
    @abstractmethod
    def validate_parameters(self) -> None:
        """
        Validate strategy parameters
        
        Raises:
            ValueError: If parameters are invalid
        """
        pass
    
    @abstractmethod
    def to_nautilus(self):
        """
        Convert strategy to Nautilus Trader format
        
        Returns:
            Nautilus Strategy instance
        """
        pass
    
    @abstractmethod
    def to_backtrader(self):
        """
        Convert strategy to Backtrader format
        
        Returns:
            Backtrader Strategy class
        """
        pass
    
    @abstractmethod
    def get_parameter_space(self) -> Dict[str, tuple]:
        """
        Get parameter optimization space
        
        Returns:
            Dictionary mapping parameter names to (min, max) tuples
        """
        pass
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """
        Get strategy information summary
        
        Returns:
            Dictionary with strategy details
        """
        return {
            'name': self.config.name,
            'framework': self.config.framework.value,
            'parameters': self.config.parameters,
            'timeframe': self.config.timeframe,
            'symbols': self.config.symbols,
            'capital': self.config.capital,
            'risk_params': self.config.risk_params
        }
