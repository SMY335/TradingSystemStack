"""
Feature Engineering for ML Trading Models
Add 50+ technical indicators for machine learning
"""

import pandas as pd
import numpy as np
import ta


class FeatureEngineer:
    """Feature engineering pour ML trading models"""
    
    @staticmethod
    def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Ajouter 50+ indicateurs techniques"""
        df = df.copy()
        
        # Ensure we have the required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Trend indicators
        df['sma_20'] = ta.trend.sma_indicator(df['close'], window=20)
        df['sma_50'] = ta.trend.sma_indicator(df['close'], window=50)
        df['ema_12'] = ta.trend.ema_indicator(df['close'], window=12)
        df['ema_26'] = ta.trend.ema_indicator(df['close'], window=26)
        df['ema_50'] = ta.trend.ema_indicator(df['close'], window=50)
        
        # MACD
        macd = ta.trend.MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
        
        # ADX
        adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'])
        df['adx'] = adx.adx()
        df['adx_pos'] = adx.adx_pos()
        df['adx_neg'] = adx.adx_neg()
        
        # Momentum indicators
        df['rsi'] = ta.momentum.rsi(df['close'], window=14)
        df['rsi_fast'] = ta.momentum.rsi(df['close'], window=7)
        df['stoch'] = ta.momentum.stoch(df['high'], df['low'], df['close'])
        df['stoch_signal'] = ta.momentum.stoch_signal(df['high'], df['low'], df['close'])
        
        # Williams %R
        df['williams_r'] = ta.momentum.williams_r(df['high'], df['low'], df['close'])
        
        # ROC
        df['roc'] = ta.momentum.roc(df['close'], window=12)
        
        # Volatility indicators
        bb = ta.volatility.BollingerBands(df['close'])
        df['bb_high'] = bb.bollinger_hband()
        df['bb_low'] = bb.bollinger_lband()
        df['bb_mid'] = bb.bollinger_mavg()
        df['bb_width'] = (df['bb_high'] - df['bb_low']) / df['bb_mid']
        df['bb_pct'] = (df['close'] - df['bb_low']) / (df['bb_high'] - df['bb_low'])
        
        # ATR
        df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'])
        
        # Keltner Channels
        kc = ta.volatility.KeltnerChannel(df['high'], df['low'], df['close'])
        df['kc_high'] = kc.keltner_channel_hband()
        df['kc_low'] = kc.keltner_channel_lband()
        df['kc_mid'] = kc.keltner_channel_mband()
        
        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # OBV (On-Balance Volume)
        df['obv'] = ta.volume.on_balance_volume(df['close'], df['volume'])
        
        # Volume Weighted Average Price
        df['vwap'] = ta.volume.volume_weighted_average_price(
            df['high'], df['low'], df['close'], df['volume']
        )
        
        # Price patterns and returns
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # High-Low range
        df['high_low_pct'] = (df['high'] - df['low']) / df['close']
        df['close_open_pct'] = (df['close'] - df['open']) / df['open']
        
        # Price position relative to high/low
        df['close_to_high'] = (df['high'] - df['close']) / df['high']
        df['close_to_low'] = (df['close'] - df['low']) / df['low']
        
        # Moving average crossovers
        df['sma_20_50_diff'] = df['sma_20'] - df['sma_50']
        df['ema_12_26_diff'] = df['ema_12'] - df['ema_26']
        
        # Price vs moving averages
        df['close_sma20_ratio'] = df['close'] / df['sma_20']
        df['close_sma50_ratio'] = df['close'] / df['sma_50']
        df['close_ema12_ratio'] = df['close'] / df['ema_12']
        
        # Lag features
        for lag in [1, 2, 3, 5, 10]:
            df[f'returns_lag_{lag}'] = df['returns'].shift(lag)
            df[f'close_lag_{lag}'] = df['close'].shift(lag)
            df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
        
        # Rolling statistics
        for window in [5, 10, 20]:
            df[f'returns_std_{window}'] = df['returns'].rolling(window=window).std()
            df[f'returns_mean_{window}'] = df['returns'].rolling(window=window).mean()
            df[f'volume_std_{window}'] = df['volume'].rolling(window=window).std()
        
        return df
    
    @staticmethod
    def create_labels(df: pd.DataFrame, forward_periods: int = 5, threshold: float = 0.02) -> pd.DataFrame:
        """Créer labels pour classification (0=down, 1=neutral, 2=up)"""
        df = df.copy()
        
        # Calculate future return
        df['future_return'] = df['close'].shift(-forward_periods) / df['close'] - 1
        
        # Create labels (0=down, 1=neutral, 2=up for XGBoost compatibility)
        df['label'] = 1  # neutral
        df.loc[df['future_return'] > threshold, 'label'] = 2  # up
        df.loc[df['future_return'] < -threshold, 'label'] = 0  # down
        
        return df
    
    @staticmethod
    def prepare_ml_dataset(df: pd.DataFrame, drop_na: bool = True) -> pd.DataFrame:
        """Prepare complete dataset for ML"""
        
        # Add technical indicators
        df = FeatureEngineer.add_technical_indicators(df)
        
        # Create labels
        df = FeatureEngineer.create_labels(df)
        
        # Drop NaN values if requested
        if drop_na:
            df = df.dropna()
        
        return df
    
    @staticmethod
    def get_feature_columns(df: pd.DataFrame) -> list:
        """Get list of feature columns (excluding OHLCV and labels)"""
        exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'label', 'future_return']
        return [col for col in df.columns if col not in exclude_cols]
