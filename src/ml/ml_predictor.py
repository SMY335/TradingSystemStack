"""
ML Predictor for Trading
Support for multiple ML models: XGBoost, LightGBM, Random Forest
"""

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import xgboost as xgb
import lightgbm as lgb
import pandas as pd
import numpy as np
import joblib
from typing import Dict, Any, Tuple


class MLPredictor:
    """ML model pour prédire direction du prix"""
    
    def __init__(self, model_type: str = 'xgboost'):
        """
        Initialize ML predictor
        
        Args:
            model_type: 'xgboost', 'lightgbm', 'random_forest', 'gradient_boosting'
        """
        self.model_type = model_type
        self.model = None
        self.feature_names = None
        
    def train(self, df: pd.DataFrame, target_col: str = 'label') -> Dict[str, Any]:
        """Entraîner le modèle"""
        
        # Préparer features
        feature_cols = [col for col in df.columns 
                       if col not in ['label', 'future_return', 'open', 'high', 'low', 'close', 'volume']]
        
        X = df[feature_cols].fillna(0)
        y = df[target_col]
        
        # Remove rows with NaN target
        valid_idx = ~y.isna()
        X = X[valid_idx]
        y = y[valid_idx]
        
        self.feature_names = feature_cols
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )
        
        print(f"📊 Training set: {len(X_train)} samples")
        print(f"📊 Test set: {len(X_test)} samples")
        print(f"📊 Features: {len(feature_cols)}")
        
        # Train model
        if self.model_type == 'xgboost':
            self.model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        elif self.model_type == 'lightgbm':
            self.model = lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
                verbose=-1
            )
        elif self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
        elif self.model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        else:
            raise ValueError(f"Model type {self.model_type} non supporté")
        
        print(f"\n🤖 Training {self.model_type}...")
        self.model.fit(X_train, y_train)
        
        # Evaluate
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        
        print(f"\n📈 Train accuracy: {train_score:.4f}")
        print(f"📈 Test accuracy: {test_score:.4f}")
        
        # Detailed report
        y_pred = self.model.predict(X_test)
        print("\n📊 Classification Report:")
        print(classification_report(y_test, y_pred))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print("\n📊 Confusion Matrix:")
        print(cm)
        
        # Feature importance
        importance_df = None
        if hasattr(self.model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False).head(10)
            
            print("\n🔝 Top 10 Features:")
            print(importance_df.to_string(index=False))
        
        return {
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'confusion_matrix': cm.tolist(),
            'feature_importance': importance_df.to_dict() if importance_df is not None else None
        }
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Prédire direction du prix"""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        X = df[self.feature_names].fillna(0)
        return self.model.predict(X)
    
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Prédire probabilités"""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        X = df[self.feature_names].fillna(0)
        return self.model.predict_proba(X)
    
    def save(self, filepath: str):
        """Sauvegarder le modèle"""
        joblib.dump({
            'model': self.model,
            'feature_names': self.feature_names,
            'model_type': self.model_type
        }, filepath)
        print(f"✅ Modèle sauvegardé: {filepath}")
    
    def load(self, filepath: str):
        """Charger le modèle"""
        data = joblib.load(filepath)
        self.model = data['model']
        self.feature_names = data['feature_names']
        self.model_type = data['model_type']
        print(f"✅ Modèle chargé: {filepath}")
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """Get feature importance"""
        if not hasattr(self.model, 'feature_importances_'):
            return None
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False).head(top_n)
        
        return importance_df
