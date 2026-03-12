"""
PROFESSIONAL ENSEMBLE ML MODELS
Hedge-fund-grade machine learning for quantitative trading
XGBoost + LightGBM + RandomForest + Neural Network
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional

# ML Models
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.neural_network import MLPRegressor

# Preprocessing
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

import warnings
warnings.filterwarnings('ignore')


class EnsembleMLSystem:
    """
    Professional ensemble ML system for return prediction
    Combines multiple models with intelligent weighting
    """
    
    def __init__(self, target_horizon: int = 5):
        """
        Initialize ensemble system
        
        Args:
            target_horizon: Days ahead to predict (default: 5)
        """
        self.target_horizon = target_horizon
        self.models = {}
        self.model_weights = {}
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_importance = None
        
        print(f"Initializing Ensemble ML System (target horizon: {target_horizon} days)")
        
    def initialize_models(self):
        """Initialize all ML models with optimized hyperparameters"""
        
        # 1. XGBoost - Powerful gradient boosting
        self.models['xgboost'] = XGBRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            tree_method='hist'
        )
        
        # 2. LightGBM - Fast gradient boosting
        self.models['lightgbm'] = LGBMRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        
        # 3. Random Forest - Robust ensemble
        self.models['random_forest'] = RandomForestRegressor(
            n_estimators=200,
            max_depth=8,
            min_samples_split=20,
            min_samples_leaf=10,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
        
        # 4. Gradient Boosting - Traditional boosting
        self.models['gradient_boosting'] = GradientBoostingRegressor(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42
        )
        
        # 5. Neural Network - Deep learning
        self.models['neural_network'] = MLPRegressor(
            hidden_layer_sizes=(100, 50, 25),
            activation='relu',
            solver='adam',
            alpha=0.01,
            batch_size='auto',
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        )
        
        print(f"Initialized {len(self.models)} models: {list(self.models.keys())}")
    
    def prepare_training_data(self, features: pd.DataFrame, 
                            prices: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and target for training
        
        Args:
            features: Feature matrix
            prices: Price series for creating target
            
        Returns:
            X, y for training
        """
        # Create target: forward returns
        returns = prices.pct_change()
        target = returns.shift(-self.target_horizon)
        
        # Align features and target
        valid_idx = features.index.intersection(target.index)
        X = features.loc[valid_idx]
        y = target.loc[valid_idx]
        
        # Remove NaN
        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]
        
        print(f"Training data shape: X={X.shape}, y={y.shape}")
        
        return X, y
    
    def train_ensemble(self, features: pd.DataFrame, 
                      prices: pd.Series,
                      validation_split: float = 0.2):
        """
        Train all models in ensemble
        
        Args:
            features: Feature matrix
            prices: Price series
            validation_split: Fraction for validation (walk-forward)
        """
        print("\n" + "="*80)
        print("TRAINING ENSEMBLE ML MODELS")
        print("="*80)
        
        # Initialize models
        if not self.models:
            self.initialize_models()
        
        # Prepare data
        X, y = self.prepare_training_data(features, prices)
        
        # Split into train/val using walk-forward
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
        
        print(f"Train size: {len(X_train)}, Validation size: {len(X_val)}")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Train each model
        val_scores = {}
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            
            try:
                # Train
                model.fit(X_train_scaled, y_train)
                
                # Validate
                val_pred = model.predict(X_val_scaled)
                mse = mean_squared_error(y_val, val_pred)
                r2 = r2_score(y_val, val_pred)
                
                # Direction accuracy
                direction_acc = np.mean(np.sign(val_pred) == np.sign(y_val))
                
                val_scores[name] = {
                    'mse': mse,
                    'r2': r2,
                    'direction_acc': direction_acc,
                    'rmse': np.sqrt(mse)
                }
                
                print(f"  MSE: {mse:.6f}")
                print(f"  R²: {r2:.4f}")
                print(f"  Direction Accuracy: {direction_acc:.2%}")
                
            except Exception as e:
                print(f"  ERROR: {e}")
                val_scores[name] = {'mse': 999, 'r2': -999, 'direction_acc': 0.5, 'rmse': 999}
        
        # Calculate model weights based on validation performance
        self._calculate_model_weights(val_scores)
        
        # Get feature importance
        self._extract_feature_importance(X.columns)
        
        self.is_fitted = True
        print("\n✓ Ensemble training complete!")
        
        return val_scores
    
    def _calculate_model_weights(self, val_scores: Dict):
        """Calculate ensemble weights based on validation performance"""
        
        # Use inverse RMSE as weights (better models get higher weight)
        rmse_values = {name: scores['rmse'] for name, scores in val_scores.items()}
        inv_rmse = {name: 1.0 / (rmse + 1e-6) for name, rmse in rmse_values.items()}
        
        # Normalize to sum to 1
        total = sum(inv_rmse.values())
        self.model_weights = {name: w / total for name, w in inv_rmse.items()}
        
        print("\nModel Weights (based on validation RMSE):")
        for name, weight in sorted(self.model_weights.items(), key=lambda x: -x[1]):
            print(f"  {name:20s}: {weight:.3f}")
    
    def _extract_feature_importance(self, feature_names):
        """Extract feature importance from tree-based models"""
        
        importance_dict = {}
        
        for name in ['xgboost', 'lightgbm', 'random_forest', 'gradient_boosting']:
            if name in self.models:
                try:
                    importance_dict[name] = self.models[name].feature_importances_
                except:
                    pass
        
        if importance_dict:
            # Average importance across models
            avg_importance = np.mean(list(importance_dict.values()), axis=0)
            self.feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': avg_importance
            }).sort_values('importance', ascending=False)
        
    def predict(self, features: pd.DataFrame) -> pd.Series:
        """
        Generate ensemble predictions
        
        Args:
            features: Feature matrix
            
        Returns:
            Predicted returns
        """
        if not self.is_fitted:
            raise ValueError("Models not trained! Call train_ensemble() first.")
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Get predictions from each model
        predictions = {}
        for name, model in self.models.items():
            try:
                predictions[name] = model.predict(features_scaled)
            except Exception as e:
                print(f"Warning: {name} prediction failed: {e}")
                predictions[name] = np.zeros(len(features))
        
        # Weighted ensemble
        ensemble_pred = np.zeros(len(features))
        for name, pred in predictions.items():
            weight = self.model_weights.get(name, 0)
            ensemble_pred += weight * pred
        
        return pd.Series(ensemble_pred, index=features.index)
    
    def predict_with_confidence(self, features: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        Generate predictions with confidence intervals
        
        Args:
            features: Feature matrix
            
        Returns:
            predictions, confidence (std of model predictions)
        """
        if not self.is_fitted:
            raise ValueError("Models not trained! Call train_ensemble() first.")
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Get predictions from each model
        all_preds = []
        for name, model in self.models.items():
            try:
                pred = model.predict(features_scaled)
                all_preds.append(pred)
            except Exception as e:
                print(f"Warning: {name} prediction failed: {e}")
        
        # Calculate mean and std
        all_preds = np.array(all_preds)
        mean_pred = np.mean(all_preds, axis=0)
        std_pred = np.std(all_preds, axis=0)
        
        predictions = pd.Series(mean_pred, index=features.index)
        confidence = pd.Series(std_pred, index=features.index)
        
        return predictions, confidence
    
    def get_top_features(self, n: int = 20) -> pd.DataFrame:
        """Get top N most important features"""
        if self.feature_importance is None:
            return pd.DataFrame()
        return self.feature_importance.head(n)
    
    def cross_validate(self, features: pd.DataFrame, 
                       prices: pd.Series,
                       n_splits: int = 5) -> Dict:
        """
        Perform time-series cross-validation
        
        Args:
            features: Feature matrix
            prices: Price series
            n_splits: Number of CV splits
            
        Returns:
            CV scores for each model
        """
        print("\n" + "="*80)
        print("CROSS-VALIDATION")
        print("="*80)
        
        if not self.models:
            self.initialize_models()
        
        X, y = self.prepare_training_data(features, prices)
        
        # Time series split
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        cv_scores = {name: [] for name in self.models.keys()}
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
            print(f"\nFold {fold}/{n_splits}")
            
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Scale
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            for name, model in self.models.items():
                try:
                    # Clone and train
                    from sklearn.base import clone
                    model_clone = clone(model)
                    model_clone.fit(X_train_scaled, y_train)
                    
                    # Predict
                    val_pred = model_clone.predict(X_val_scaled)
                    
                    # Score
                    r2 = r2_score(y_val, val_pred)
                    cv_scores[name].append(r2)
                    
                except Exception as e:
                    print(f"  {name}: ERROR - {e}")
                    cv_scores[name].append(-999)
        
        # Summary
        print("\n" + "="*80)
        print("CV RESULTS (R² Score)")
        print("="*80)
        cv_summary = {}
        for name, scores in cv_scores.items():
            valid_scores = [s for s in scores if s > -999]
            if valid_scores:
                cv_summary[name] = {
                    'mean': np.mean(valid_scores),
                    'std': np.std(valid_scores),
                    'min': np.min(valid_scores),
                    'max': np.max(valid_scores)
                }
                print(f"{name:20s}: {cv_summary[name]['mean']:.4f} ± {cv_summary[name]['std']:.4f}")
        
        return cv_summary


if __name__ == "__main__":
    # Example usage
    import yfinance as yf
    from PRO_feature_engineering import AdvancedFeatureEngineer
    
    # Download data
    data = yf.download('ICLN', start='2019-01-01', end='2024-12-31')
    
    # Create features
    engineer = AdvancedFeatureEngineer()
    features = engineer.create_all_features(data, data['Volume'])
    
    # Initialize ensemble
    ensemble = EnsembleMLSystem(target_horizon=5)
    
    # Train
    val_scores = ensemble.train_ensemble(features, data['Close'])
    
    # Predict
    predictions = ensemble.predict(features)
    
    print(f"\nPredictions shape: {predictions.shape}")
    print(f"Predictions range: [{predictions.min():.4f}, {predictions.max():.4f}]")
