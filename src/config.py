import logging
import os

# Setup logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Get current directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

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

# Forecasting Parameters
FORECAST_PERIODS = 30
TRAIN_TEST_SPLIT = 0.8
LAG_PERIODS = [1, 7, 14, 30]
ROLLING_WINDOWS = [7, 14, 30]

# Inventory Optimization
STOCK_BUFFER = 1.2
SAFETY_STOCK = 0.3
REORDER_POINT = 0.7

# Output Files
FORECAST_PLOT = 'sales_forecast.png'
FORECAST_CSV = 'forecast_results.csv'
OPPORTUNITIES_CSV = 'revenue_opportunities.csv'
FEATURE_IMPORTANCE_CSV = 'feature_importance.csv'
METRICS_JSON = 'model_metrics.json'