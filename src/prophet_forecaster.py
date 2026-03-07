import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from config import logger, PROPHET_PARAMS, FORECAST_PERIODS
from utils import save_plot, save_dataframe, calculate_metrics

class ProphetForecaster:
    """Time series forecasting using Facebook Prophet"""
    
    def __init__(self):
        self.model = None
        self.forecast = None
        self.metrics = None
        logger.info("ProphetForecaster initialized")
    
    def prepare_data(self, df, date_col='date', target_col='revenue'):
        """Prepare data in Prophet format"""
        prophet_df = df[[date_col, target_col]].copy()
        prophet_df.columns = ['ds', 'y']
        prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])
        return prophet_df
    
    def train(self, df):
        """Train Prophet model"""
        logger.info("Training Prophet model")
        
        self.model = Prophet(
            yearly_seasonality=PROPHET_PARAMS['yearly_seasonality'],
            weekly_seasonality=PROPHET_PARAMS['weekly_seasonality'],
            daily_seasonality=PROPHET_PARAMS['daily_seasonality'],
            seasonality_mode=PROPHET_PARAMS['seasonality_mode'],
            changepoint_prior_scale=PROPHET_PARAMS['changepoint_prior_scale']
        )
        
        # Add custom seasonalities
        self.model.add_seasonality(
            name='monthly',
            period=30.5,
            fourier_order=5
        )
        
        self.model.fit(df)
        logger.info("Prophet model trained successfully")
    
    def predict(self, periods=FORECAST_PERIODS):
        """Generate forecast"""
        logger.info(f"Generating {periods}-day forecast")
        
        future = self.model.make_future_dataframe(periods=periods)
        self.forecast = self.model.predict(future)
        
        forecast_df = self.forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
        forecast_df['ds'] = forecast_df['ds'].dt.strftime('%Y-%m-%d')
        
        return forecast_df
    
    def evaluate(self, test_df):
        """Evaluate model performance"""
        logger.info("Evaluating Prophet model")
        
        # Predict on test set
        forecast = self.model.predict(test_df[['ds']])
        
        # Calculate metrics
        self.metrics = calculate_metrics(test_df['y'], forecast['yhat'])
        
        logger.info(f"Model Accuracy: {self.metrics['accuracy']}%")
        return self.metrics
    
    def plot_forecast(self, save_path=None):
        """Plot forecast results"""
        if self.forecast is None:
            logger.error("No forecast available. Run predict() first.")
            return None
        
        fig = self.model.plot(self.forecast, figsize=(12, 6))
        plt.title('Sales Forecast - Prophet Model', fontsize=14, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('Revenue')
        plt.tight_layout()
        
        if save_path:
            save_plot(fig, save_path)
        
        return fig
    
    def plot_components(self, save_path=None):
        """Plot forecast components (trend, weekly, yearly)"""
        if self.forecast is None:
            logger.error("No forecast available. Run predict() first.")
            return None
        
        fig = self.model.plot_components(self.forecast, figsize=(12, 8))
        plt.tight_layout()
        
        if save_path:
            save_plot(fig, save_path)
        
        return fig
    
    def save_forecast(self, filepath):
        """Save forecast to CSV"""
        if self.forecast is not None:
            forecast_df = self.forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
            forecast_df['ds'] = forecast_df['ds'].dt.strftime('%Y-%m-%d')
            save_dataframe(forecast_df, filepath)
            return filepath
        else:
            logger.error("No forecast to save")
            return None