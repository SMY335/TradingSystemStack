"""
ML Integration Tests
Test feature engineering and ML predictor functionality
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from src.ml.feature_engineering import FeatureEngineer
from src.ml.ml_predictor import MLPredictor


def test_feature_engineering():
    """Test feature engineering"""
    print("\n" + "="*60)
    print("TEST 1: Feature Engineering")
    print("="*60)
    
    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=500, freq='1h')
    df = pd.DataFrame({
        'open': np.random.uniform(100, 150, 500),
        'high': np.random.uniform(100, 150, 500),
        'low': np.random.uniform(100, 150, 500),
        'close': np.random.uniform(100, 150, 500),
        'volume': np.random.uniform(1000, 10000, 500)
    }, index=dates)
    
    # Make high >= close >= low
    df['high'] = df[['open', 'high', 'close']].max(axis=1) * 1.01
    df['low'] = df[['open', 'low', 'close']].min(axis=1) * 0.99
    
    print(f"Original data shape: {df.shape}")
    
    engineer = FeatureEngineer()
    df_features = engineer.add_technical_indicators(df)
    df_features = engineer.create_labels(df_features)
    
    assert 'rsi' in df_features.columns, "RSI indicator not found"
    assert 'macd' in df_features.columns, "MACD indicator not found"
    assert 'label' in df_features.columns, "Label not found"
    assert 'bb_high' in df_features.columns, "Bollinger Band High not found"
    
    print(f"✅ Features created: {len(df_features.columns)} columns")
    
    # Get feature columns
    feature_cols = engineer.get_feature_columns(df_features)
    print(f"✅ Feature columns: {len(feature_cols)}")
    print(f"   Sample features: {feature_cols[:10]}")
    
    # Check label distribution
    label_dist = df_features['label'].value_counts()
    print(f"\n📊 Label distribution:")
    print(label_dist)
    
    return True


def test_ml_training():
    """Test ML training"""
    print("\n" + "="*60)
    print("TEST 2: ML Model Training")
    print("="*60)
    
    # Generate data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=500, freq='1h')
    
    # Create more realistic price data with trend
    close_prices = 100 + np.cumsum(np.random.randn(500) * 0.5)
    df = pd.DataFrame({
        'open': close_prices + np.random.randn(500) * 0.5,
        'high': close_prices + np.abs(np.random.randn(500)) * 1.0,
        'low': close_prices - np.abs(np.random.randn(500)) * 1.0,
        'close': close_prices,
        'volume': np.random.uniform(1000, 10000, 500)
    }, index=dates)
    
    engineer = FeatureEngineer()
    df = engineer.prepare_ml_dataset(df, drop_na=True)
    
    print(f"Dataset shape after feature engineering: {df.shape}")
    
    # Train multiple models
    models = ['random_forest', 'xgboost', 'lightgbm']
    results = {}
    
    for model_type in models:
        print(f"\n{'='*60}")
        print(f"Training {model_type.upper()}")
        print(f"{'='*60}")
        
        predictor = MLPredictor(model_type=model_type)
        metrics = predictor.train(df)
        
        assert metrics['test_accuracy'] > 0, f"{model_type} has zero accuracy"
        print(f"\n✅ {model_type} trained successfully")
        print(f"   Test accuracy: {metrics['test_accuracy']:.4f}")
        
        # Test predictions
        predictions = predictor.predict(df.head(10))
        assert len(predictions) == 10, "Incorrect number of predictions"
        print(f"✅ Predictions: {predictions}")
        
        # Test probabilities
        probas = predictor.predict_proba(df.head(10))
        assert probas.shape[0] == 10, "Incorrect probability shape"
        print(f"✅ Probabilities shape: {probas.shape}")
        
        results[model_type] = metrics
    
    # Compare models
    print(f"\n{'='*60}")
    print("MODEL COMPARISON")
    print(f"{'='*60}")
    for model_type, metrics in results.items():
        print(f"{model_type:20s}: {metrics['test_accuracy']:.4f}")
    
    return True


def test_model_save_load():
    """Test model save and load"""
    print("\n" + "="*60)
    print("TEST 3: Model Save/Load")
    print("="*60)
    
    # Generate data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=300, freq='1h')
    close_prices = 100 + np.cumsum(np.random.randn(300) * 0.5)
    df = pd.DataFrame({
        'open': close_prices + np.random.randn(300) * 0.5,
        'high': close_prices + np.abs(np.random.randn(300)) * 1.0,
        'low': close_prices - np.abs(np.random.randn(300)) * 1.0,
        'close': close_prices,
        'volume': np.random.uniform(1000, 10000, 300)
    }, index=dates)
    
    engineer = FeatureEngineer()
    df = engineer.prepare_ml_dataset(df, drop_na=True)
    
    # Train model
    predictor = MLPredictor(model_type='random_forest')
    predictor.train(df)
    
    # Save model
    import tempfile
    import os
    
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, 'test_model.pkl')
        predictor.save(model_path)
        
        assert os.path.exists(model_path), "Model file not created"
        print(f"✅ Model saved to {model_path}")
        
        # Load model
        predictor2 = MLPredictor()
        predictor2.load(model_path)
        
        assert predictor2.model is not None, "Model not loaded"
        assert predictor2.feature_names == predictor.feature_names, "Feature names mismatch"
        print(f"✅ Model loaded successfully")
        
        # Test predictions with loaded model
        pred1 = predictor.predict(df.head(5))
        pred2 = predictor2.predict(df.head(5))
        
        assert np.array_equal(pred1, pred2), "Predictions don't match"
        print(f"✅ Predictions match after reload")
    
    return True


if __name__ == "__main__":
    print("\n" + "="*60)
    print("🧪 ML INTEGRATION TESTS")
    print("="*60)
    
    try:
        test_feature_engineering()
        test_ml_training()
        test_model_save_load()
        
        print("\n" + "="*60)
        print("🎉 ALL TESTS PASSED!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
