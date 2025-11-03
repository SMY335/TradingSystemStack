"""
Strategy Factory
Creates strategy instances for different frameworks using adapters
"""
from typing import Dict, Any
from src.adapters.base_strategy_adapter import (
    BaseStrategyAdapter,
    StrategyConfig,
    StrategyFramework
)
from src.adapters.ema_adapter import EMAAdapter
from src.adapters.rsi_adapter import RSIAdapter
from src.adapters.macd_adapter import MACDAdapter


class StrategyFactory:
    """
    Factory for creating strategies on any framework
    Provides a unified interface for strategy instantiation
    """
    
    # Registry of available adapters
    ADAPTERS: Dict[str, type[BaseStrategyAdapter]] = {
        'ema': EMAAdapter,
        'rsi': RSIAdapter,
        'macd': MACDAdapter,
    }
    
    @staticmethod
    def create(strategy_name: str, config: StrategyConfig):
        """
        Create a strategy instance using the specified framework
        
        Args:
            strategy_name: Name of the strategy (e.g., 'ema', 'rsi', 'macd')
            config: Strategy configuration
            
        Returns:
            Strategy instance for the specified framework
            
        Raises:
            ValueError: If strategy name or framework is not supported
        """
        # Validate strategy name
        if strategy_name not in StrategyFactory.ADAPTERS:
            available = ', '.join(StrategyFactory.ADAPTERS.keys())
            raise ValueError(
                f"Unknown strategy: {strategy_name}. "
                f"Available strategies: {available}"
            )
        
        # Get adapter class and instantiate
        adapter_class = StrategyFactory.ADAPTERS[strategy_name]
        adapter = adapter_class(config)
        
        # Convert to appropriate framework
        if config.framework == StrategyFramework.NAUTILUS:
            return adapter.to_nautilus()
        elif config.framework == StrategyFramework.BACKTRADER:
            return adapter.to_backtrader()
        elif config.framework == StrategyFramework.VECTORBT:
            raise NotImplementedError(
                "VectorBT framework conversion not yet implemented. "
                "Use NAUTILUS or BACKTRADER instead."
            )
        else:
            raise ValueError(f"Unsupported framework: {config.framework}")
    
    @staticmethod
    def get_adapter(strategy_name: str, config: StrategyConfig) -> BaseStrategyAdapter:
        """
        Get the adapter instance without converting to specific framework
        Useful for accessing adapter methods like get_parameter_space()
        
        Args:
            strategy_name: Name of the strategy
            config: Strategy configuration
            
        Returns:
            Strategy adapter instance
        """
        if strategy_name not in StrategyFactory.ADAPTERS:
            available = ', '.join(StrategyFactory.ADAPTERS.keys())
            raise ValueError(
                f"Unknown strategy: {strategy_name}. "
                f"Available strategies: {available}"
            )
        
        adapter_class = StrategyFactory.ADAPTERS[strategy_name]
        return adapter_class(config)
    
    @staticmethod
    def list_strategies() -> list[str]:
        """
        List all available strategies
        
        Returns:
            List of strategy names
        """
        return list(StrategyFactory.ADAPTERS.keys())
    
    @staticmethod
    def get_strategy_info(strategy_name: str) -> Dict[str, Any]:
        """
        Get information about a strategy
        
        Args:
            strategy_name: Name of the strategy
            
        Returns:
            Dictionary with strategy information
        """
        if strategy_name not in StrategyFactory.ADAPTERS:
            raise ValueError(f"Unknown strategy: {strategy_name}")
        
        adapter_class = StrategyFactory.ADAPTERS[strategy_name]
        
        # Create a temporary config to get info
        temp_config = StrategyConfig(
            name=strategy_name,
            parameters={},
            timeframe='1h',
            symbols=['BTC/USDT'],
            capital=10000.0,
            framework=StrategyFramework.NAUTILUS
        )
        
        try:
            adapter = adapter_class(temp_config)
            return {
                'name': strategy_name,
                'adapter_class': adapter_class.__name__,
                'parameter_space': adapter.get_parameter_space(),
                'supported_frameworks': ['nautilus', 'backtrader']
            }
        except Exception:
            # If validation fails with empty params, return basic info
            return {
                'name': strategy_name,
                'adapter_class': adapter_class.__name__,
                'supported_frameworks': ['nautilus', 'backtrader']
            }
    
    @staticmethod
    def register_adapter(name: str, adapter_class: type[BaseStrategyAdapter]):
        """
        Register a new strategy adapter
        Allows extending the factory with custom strategies
        
        Args:
            name: Strategy name
            adapter_class: Adapter class (must inherit from BaseStrategyAdapter)
        """
        if not issubclass(adapter_class, BaseStrategyAdapter):
            raise ValueError(
                f"Adapter class must inherit from BaseStrategyAdapter"
            )
        
        StrategyFactory.ADAPTERS[name] = adapter_class
        print(f"✅ Registered strategy adapter: {name}")


# Convenience functions for common use cases

def create_ema_strategy(
    fast_period: int = 10,
    slow_period: int = 50,
    symbols: list[str] = None,
    timeframe: str = '1h',
    capital: float = 10000.0,
    framework: StrategyFramework = StrategyFramework.NAUTILUS
):
    """
    Quick creation of EMA strategy
    
    Args:
        fast_period: Fast EMA period
        slow_period: Slow EMA period
        symbols: Trading symbols
        timeframe: Timeframe
        capital: Initial capital
        framework: Trading framework
        
    Returns:
        EMA strategy instance
    """
    if symbols is None:
        symbols = ['BTC/USDT']
    
    config = StrategyConfig(
        name='ema_crossover',
        parameters={'fast_period': fast_period, 'slow_period': slow_period},
        timeframe=timeframe,
        symbols=symbols,
        capital=capital,
        framework=framework
    )
    
    return StrategyFactory.create('ema', config)


def create_rsi_strategy(
    period: int = 14,
    oversold: int = 30,
    overbought: int = 70,
    symbols: list[str] = None,
    timeframe: str = '1h',
    capital: float = 10000.0,
    framework: StrategyFramework = StrategyFramework.NAUTILUS
):
    """
    Quick creation of RSI strategy
    
    Args:
        period: RSI period
        oversold: Oversold threshold
        overbought: Overbought threshold
        symbols: Trading symbols
        timeframe: Timeframe
        capital: Initial capital
        framework: Trading framework
        
    Returns:
        RSI strategy instance
    """
    if symbols is None:
        symbols = ['BTC/USDT']
    
    config = StrategyConfig(
        name='rsi_strategy',
        parameters={'period': period, 'oversold': oversold, 'overbought': overbought},
        timeframe=timeframe,
        symbols=symbols,
        capital=capital,
        framework=framework
    )
    
    return StrategyFactory.create('rsi', config)


def create_macd_strategy(
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
    symbols: list[str] = None,
    timeframe: str = '1h',
    capital: float = 10000.0,
    framework: StrategyFramework = StrategyFramework.NAUTILUS
):
    """
    Quick creation of MACD strategy
    
    Args:
        fast_period: Fast EMA period
        slow_period: Slow EMA period
        signal_period: Signal line period
        symbols: Trading symbols
        timeframe: Timeframe
        capital: Initial capital
        framework: Trading framework
        
    Returns:
        MACD strategy instance
    """
    if symbols is None:
        symbols = ['BTC/USDT']
    
    config = StrategyConfig(
        name='macd_strategy',
        parameters={
            'fast_period': fast_period,
            'slow_period': slow_period,
            'signal_period': signal_period
        },
        timeframe=timeframe,
        symbols=symbols,
        capital=capital,
        framework=framework
    )
    
    return StrategyFactory.create('macd', config)


if __name__ == "__main__":
    # Example usage
    print("Available strategies:", StrategyFactory.list_strategies())
    
    # Get info about each strategy
    for strategy_name in StrategyFactory.list_strategies():
        print(f"\n{strategy_name.upper()} Strategy Info:")
        info = StrategyFactory.get_strategy_info(strategy_name)
        for key, value in info.items():
            print(f"  {key}: {value}")
