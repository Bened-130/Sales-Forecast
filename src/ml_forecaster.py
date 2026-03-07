import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from config import logger, RANDOM_FOREST_PARAMS, GRADIENT_BOOSTING_PARAMS, LAG_PERIODS, ROLLING_WINDOWS
from utils import save_dataframe, calculate_metrics

class MLForecaster:
    """Machine Learning based forecasting"""
    
    def __init__(self, model_type='random_forest'):
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.metrics = None
        logger.info(f"MLForecaster initialized with {model_type}")
    
    def create_features(self, df):
        """Create features for ML model"""
        df = df.copy()
        
        # Ensure date is datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Time-based features
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['dayofweek'] = df['date'].dt.dayofweek
        df['quarter'] = df['date'].dt.quarter
        df['dayofyear'] = df['date'].dt.dayofyear
        df['weekofyear'] = df['date'].dt.isocalendar().week.astype(int)
        
        # Lag features
        for lag in LAG_PERIODS:
            df[f'lag_{lag}'] = df['revenue'].shift(lag)
        
        # Rolling statistics
        for window in ROLLING_WINDOWS:
            df[f'rolling_mean_{window}'] = df['revenue'].rolling(window=window).mean()
            df[f'rolling_std_{window}'] = df['revenue'].rolling(window=window).std()
        
        # Drop NaN values created by lag and rolling
        df = df.dropna()
        
        return df
    
    def prepare_train_test_split(self, df, target_col='revenue', test_size=0.2):
        """Prepare features and split data"""
        feature_cols = [col for col in df.columns if col not in ['date', target_col]]
        self.feature_names = feature_cols
        
        X = df[feature_cols]
        y = df[target_col]
        
        # Time-based split (no shuffling for time series)
        split_idx = int(len(df) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        return X_train, X_test, y_train, y_test
    
    def train(self, X_train, y_train):
        """Train ML model"""
        logger.info(f"Training {self.model_type} model")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Initialize model
        if self.model_type == 'random_forest':
            self.model = RandomForestRegressor(**RANDOM_FOREST_PARAMS)
        elif self.model_type == 'gradient_boosting':
            self.model = GradientBoostingRegressor(**GRADIENT_BOOSTING_PARAMS)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Cross-validation
        cv_scores = cross_val_score(
            self.model, X_train_scaled, y_train, 
            cv=5, scoring='neg_mean_absolute_error'
        )
        
        logger.info(f"Cross-validation MAE: {-cv_scores.mean():.2f} (+/- {cv_scores.std():.2f})")
    
    def predict(self, X_test):
        """Make predictions"""
        X_test_scaled = self.scaler.transform(X_test)
        predictions = self.model.predict(X_test_scaled)
        return predictions
    
    def evaluate(self, X_test, y_test):
        """Evaluate model"""
        logger.info("Evaluating ML model")
        
        predictions = self.predict(X_test)
        
        self.metrics = calculate_metrics(y_test, predictions)
        
        logger.info(f"Model Accuracy: {self.metrics['accuracy']}%")
        return self.metrics
    
    def feature_importance(self):
        """Get feature importance"""
        if self.model is None:
            logger.error("Model not trained yet")
            return None
        
        if hasattr(self.model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            return importance_df
        return None
    
    def save_feature_importance(self, filepath):
        """Save feature importance to CSV"""
        importance_df = self.feature_importance()
        if importance_df is not None:
            save_dataframe(importance_df, filepath)
            return filepath
        return None
    
    def get_model_summary(self):
        """Get model summary statistics"""
        summary = {
            'model_type': self.model_type,
            'n_features': len(self.feature_names) if self.feature_names else 0,
            'metrics': self.metrics
        }
        return summary