"""
Configuration file for Sales Forecasting Pipeline
Contains all constants, parameters, and settings
"""

import logging
from datetime import datetime

# Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Data Generation Parameters
DATA_PARAMS = {
    'start_date': '2022-01-01',
    'end_date': '2023-12-31',
    'products': ['Coca-Cola', 'Sprite', 'Fanta', 'Dasani Water', 'Minute Maid'],
    'regions': ['Nairobi', 'Mombasa', 'Kisumu', 'Nakuru', 'Eldoret'],
    'n_distribution_points': 200,
    'base_quantity_min': 500,
    'base_quantity_max': 2000,
    'unit_price_min': 30,
    'unit_price_max': 80
}

# Model Parameters
PROPHET_PARAMS = {
    'yearly_seasonality': True,
    'weekly_seasonality': True,
    'daily_seasonality': False,
    'seasonality_mode': 'multiplicative',
    'changepoint_prior_scale': 0.05
}

RANDOM_FOREST_PARAMS = {
    'n_estimators': 100,
    'max_depth': 10,
    'random_state': 42,
    'n_jobs': -1
}

GRADIENT_BOOSTING_PARAMS = {
    'n_estimators': 100,
    'max_depth': 5,
    'learning_rate': 0.1,
    'random_state': 42
}

# Forecasting Parameters
FORECAST_PERIODS = 30  # Days to forecast
TRAIN_TEST_SPLIT = 0.8  # 80% training, 20% testing

# Feature Engineering
LAG_PERIODS = [1, 7, 14, 30]
ROLLING_WINDOWS = [7, 14, 30]

# Inventory Optimization
STOCK_BUFFER = 1.2      # 20% buffer
SAFETY_STOCK = 0.3      # 30% safety stock
REORDER_POINT = 0.7     # Reorder at 70%

# Output Paths
OUTPUT_DIR = 'output'
FORECAST_PLOT = f'{OUTPUT_DIR}/sales_forecast.png'
FORECAST_CSV = f'{OUTPUT_DIR}/forecast_results.csv'
OPPORTUNITIES_CSV = f'{OUTPUT_DIR}/revenue_opportunities.csv'
FEATURE_IMPORTANCE_CSV = f'{OUTPUT_DIR}/feature_importance.csv'
METRICS_JSON = f'{OUTPUT_DIR}/model_metrics.json'