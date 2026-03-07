import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from config import logger, OUTPUT_DIR

def ensure_output_dir():
    """Create output directory if it doesn't exist"""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        logger.info(f"Created output directory: {OUTPUT_DIR}")

def save_dataframe(df, filepath, index=False):
    """Save DataFrame to CSV"""
    ensure_output_dir()
    df.to_csv(filepath, index=index)
    logger.info(f"Saved DataFrame to {filepath}")

def save_json(data, filepath):
    """Save dictionary to JSON"""
    ensure_output_dir()
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)
    logger.info(f"Saved JSON to {filepath}")

def save_plot(fig, filepath):
    """Save matplotlib figure"""
    ensure_output_dir()
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    logger.info(f"Saved plot to {filepath}")
    plt.close(fig)

def load_dataframe(filepath):
    """Load DataFrame from CSV"""
    if os.path.exists(filepath):
        return pd.read_csv(filepath)
    else:
        logger.error(f"File not found: {filepath}")
        return None

def format_currency(value):
    """Format number as currency"""
    return f"${value:,.2f}"

def calculate_metrics(y_true, y_pred):
    """Calculate MAE, RMSE, MAPE, Accuracy"""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
    import numpy as np
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred)
    accuracy = (1 - mape) * 100
    
    return {
        'mae': round(mae, 2),
        'rmse': round(rmse, 2),
        'mape': round(mape * 100, 2),
        'accuracy': round(accuracy, 2)
    }