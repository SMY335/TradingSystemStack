# Machine Learning Integration

## Overview

The ML Integration module provides comprehensive feature engineering and machine learning capabilities for trading strategy development. It supports multiple ML models (XGBoost, LightGBM, Random Forest, Gradient Boosting) with 65+ technical indicators for price prediction.

## Features

- **Feature Engineering**: 65+ technical indicators automatically generated
- **Multiple ML Models**: XGBoost, LightGBM, Random Forest, Gradient Boosting
- **Label Generation**: Multi-class classification (Down, Neutral, Up)
- **Model Persistence**: Save/load trained models
- **Feature Importance**: Identify most predictive indicators
- **Cross-validation**: Train/test splitting with temporal awareness

## Architecture

```
Raw OHLCV Data
    ↓
FeatureEngineer.add_technical_indicators()
    ↓
65+ Technical Features
    ↓
FeatureEngineer.create_labels()
    ↓
Labeled Dataset (0=down, 1=neutral, 2=up)
    ↓
MLPredictor.train()
    ↓
Trained Model (XGBoost/LightGBM/RF/GB)
    ↓
Predictions & Probabilities
```

## Feature Engineering

### Available Indicators (65+)

#### Trend Indicators
- SMA (20, 50 periods)
- EMA (12, 26, 50 periods)
- MACD (line, signal, divergence)
- ADX (with positive/negative directional indicators)

#### Momentum Indicators
- RSI (14 period, fast 7 period)
- Stochastic (with signal)
- Williams %R
- ROC (Rate of Change)

#### Volatility Indicators
- Bollinger Bands (high, low, middle, width, percentage)
- ATR (Average True Range)
- Keltner Channels (high, low, middle)

#### Volume Indicators
- Volume SMA and ratio
- OBV (On-Balance Volume)
- VWAP (Volume Weighted Average Price)

#### Price Patterns
- Returns (simple and log)
- High-Low range percentage
- Close-Open percentage
- Price position relative to high/low

#### Lag Features
- Returns lags (1, 2, 3, 5, 10 periods)
- Close price lags (1, 2, 3, 5, 10 periods)
- Volume lags (1, 2, 3, 5, 10 periods)

#### Rolling Statistics
- Returns std/mean (5, 10, 20 periods)
- Volume std (5, 10, 20 periods)

### Usage

```python
from src.ml.feature_engineering import FeatureEngineer
import pandas as pd

# Your OHLCV data
df = pd.DataFrame({
    'open': [...],
    'high': [...],
    'low': [...],
    'close': [...],
    'volume': [...]
})

# Initialize engineer
engineer = FeatureEngineer()

# Add technical indicators
df_features = engineer.add_technical_indicators(df)

# Create labels (0=down, 1=neutral, 2=up)
df_labeled = engineer.create_labels(
    df_features,
    forward_periods=5,  # Look 5 periods ahead
    threshold=0.02      # 2% threshold for up/down
)

# Or use all-in-one
df_ready = engineer.prepare_ml_dataset(df, drop_na=True)

# Get feature column names
features = engineer.get_feature_columns(df_ready)
print(f"Total features: {len(features)}")
```

## ML Predictor

### Supported Models

1. **XGBoost** (Default)
   - Fast gradient boosting
   - Excellent performance
   - Built-in regularization

2. **LightGBM**
   - Very fast training
   - Memory efficient
   - Good for large datasets

3. **Random Forest**
   - Robust to overfitting
   - Good feature importance
   - Stable predictions

4. **Gradient Boosting**
   - Strong baseline
   - Interpretable
   - Reliable

### Training a Model

```python
from src.ml.ml_predictor import MLPredictor
from src.ml.feature_engineering import FeatureEngineer

# Prepare data
engineer = FeatureEngineer()
df = engineer.prepare_ml_dataset(raw_data, drop_na=True)

# Train XGBoost model
predictor = MLPredictor(model_type='xgboost')
metrics = predictor.train(df, target_col='label')

# View metrics
print(f"Train Accuracy: {metrics['train_accuracy']:.4f}")
print(f"Test Accuracy: {metrics['test_accuracy']:.4f}")

# Make predictions
predictions = predictor.predict(df_new)
probabilities = predictor.predict_proba(df_new)

# Save model
predictor.save('models/my_xgb_model.pkl')
```

### Model Comparison

```python
from src.ml.ml_predictor import MLPredictor

models = ['xgboost', 'lightgbm', 'random_forest', 'gradient_boosting']
results = {}

for model_type in models:
    predictor = MLPredictor(model_type=model_type)
    metrics = predictor.train(df)
    results[model_type] = metrics['test_accuracy']
    
# Compare
for model, accuracy in sorted(results.items(), key=lambda x: x[1], reverse=True):
    print(f"{model:20s}: {accuracy:.4f}")
```

## Label Generation

Labels are created based on future price movement:

- **Label 0 (Down)**: Future return < -threshold (default -2%)
- **Label 1 (Neutral)**: -threshold ≤ Future return ≤ threshold
- **Label 2 (Up)**: Future return > threshold (default 2%)

```python
df_labeled = engineer.create_labels(
    df,
    forward_periods=5,  # Predict 5 periods ahead
    threshold=0.02      # 2% threshold
)

# Check label distribution
print(df_labeled['label'].value_counts())
```

## Model Persistence

### Save Model

```python
predictor = MLPredictor(model_type='xgboost')
predictor.train(df)
predictor.save('models/btc_xgb_5h.pkl')
```

### Load Model

```python
predictor = MLPredictor()
predictor.load('models/btc_xgb_5h.pkl')

# Make predictions
predictions = predictor.predict(new_data)
```

## Feature Importance

```python
# Get top 20 features
importance_df = predictor.get_feature_importance(top_n=20)
print(importance_df)

# Visualize (if using matplotlib)
import matplotlib.pyplot as plt

importance_df.plot(
    x='feature',
    y='importance',
    kind='barh',
    figsize=(10, 8)
)
plt.title('Top 20 Feature Importance')
plt.show()
```

## Performance Metrics

The system provides comprehensive metrics:

- **Accuracy**: Overall prediction accuracy
- **Precision/Recall/F1**: Per-class metrics
- **Confusion Matrix**: Detailed error analysis
- **Feature Importance**: Most predictive features

```python
metrics = predictor.train(df)

print("Classification Report:")
print(metrics['classification_report'])

print("\nConfusion Matrix:")
print(metrics['confusion_matrix'])

print("\nFeature Importance:")
print(metrics['feature_importance'])
```

## Integration with Strategies

### Use ML Predictions in Trading

```python
from src.ml.ml_predictor import MLPredictor
from src.ml.feature_engineering import FeatureEngineer

class MLEnhancedStrategy(bt.Strategy):
    def __init__(self):
        self.engineer = FeatureEngineer()
        self.predictor = MLPredictor()
        self.predictor.load('models/trained_model.pkl')
        
    def next(self):
        # Get recent data
        recent_data = self.get_recent_ohlcv(100)  # Last 100 bars
        
        # Prepare features
        df = self.engineer.prepare_ml_dataset(recent_data)
        
        # Get prediction
        prediction = self.predictor.predict(df.tail(1))
        proba = self.predictor.predict_proba(df.tail(1))
        
        # Trade based on prediction
        if prediction[0] == 2 and proba[0][2] > 0.7:  # High confidence UP
            self.buy()
        elif prediction[0] == 0 and proba[0][0] > 0.7:  # High confidence DOWN
            self.sell()
```

## Best Practices

### Data Preparation

1. **Sufficient History**: Use at least 500+ data points
2. **Clean Data**: Remove gaps and anomalies
3. **Time-based Split**: Never shuffle time-series data
4. **Feature Scaling**: Some models benefit from normalization

### Model Training

1. **Avoid Overfitting**: Monitor train vs test accuracy gap
2. **Cross-validation**: Use walk-forward validation
3. **Feature Selection**: Remove low-importance features
4. **Hyperparameter Tuning**: Use grid search or Optuna

### Prediction

1. **Confidence Threshold**: Only act on high-confidence predictions
2. **Model Ensemble**: Combine multiple models
3. **Regular Retraining**: Update models with new data
4. **Monitoring**: Track prediction accuracy in production

## Example: Complete Workflow

```python
import pandas as pd
from src.ml.feature_engineering import FeatureEngineer
from src.ml.ml_predictor import MLPredictor

# 1. Load data
df = pd.read_csv('btc_1h_data.csv')

# 2. Prepare features
engineer = FeatureEngineer()
df_ml = engineer.prepare_ml_dataset(df, drop_na=True)

print(f"Dataset shape: {df_ml.shape}")
print(f"Features: {len(engineer.get_feature_columns(df_ml))}")

# 3. Train model
predictor = MLPredictor(model_type='xgboost')
metrics = predictor.train(df_ml)

print(f"\nTest Accuracy: {metrics['test_accuracy']:.4f}")

# 4. Feature importance
importance = predictor.get_feature_importance(top_n=10)
print("\nTop 10 Features:")
print(importance)

# 5. Save model
predictor.save('models/btc_1h_xgb.pkl')

# 6. Load and predict
predictor_new = MLPredictor()
predictor_new.load('models/btc_1h_xgb.pkl')

new_predictions = predictor_new.predict(df_ml.tail(10))
print(f"\nRecent predictions: {new_predictions}")
```

## Testing

Run ML integration tests:

```bash
python tests/integration/test_ml_integration.py
```

Tests cover:
- Feature engineering with 65+ indicators
- Training all model types
- Model save/load functionality
- Prediction consistency

## Optimization Tips

### Speed Improvements

1. **LightGBM**: Fastest for large datasets
2. **Feature Selection**: Remove low-importance features
3. **Batch Prediction**: Predict multiple samples at once
4. **Model Caching**: Load model once, reuse predictions

### Accuracy Improvements

1. **More Data**: 1000+ samples recommended
2. **Feature Engineering**: Add domain-specific features
3. **Ensemble Methods**: Combine multiple models
4. **Hyperparameter Tuning**: Use Optuna
5. **Class Balancing**: Address imbalanced labels

## Troubleshooting

### Low Accuracy

- Check label distribution (should be balanced)
- Increase training data
- Try different models
- Tune hyperparameters
- Add more features

### Overfitting

- Reduce model complexity (max_depth, n_estimators)
- Add regularization
- Use more training data
- Implement cross-validation

### Memory Issues

- Use LightGBM instead of XGBoost
- Reduce feature count
- Process data in batches
- Use float32 instead of float64

## Future Enhancements

- [ ] Deep learning models (LSTM, Transformers)
- [ ] AutoML integration
- [ ] Feature selection algorithms
- [ ] Online learning capabilities
- [ ] Multi-asset predictions
- [ ] Regime detection
- [ ] Advanced ensemble methods

## References

- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [scikit-learn Documentation](https://scikit-learn.org/)
- [TA Library](https://technical-analysis-library-in-python.readthedocs.io/)

## Support

For ML-related issues:
- Review test examples
- Check feature engineering output
- Validate data quality
- Monitor model metrics
