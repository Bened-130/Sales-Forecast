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
    return OUTPUT_DIR

def get_output_path(filename):
    """Get full path for output file"""
    ensure_output_dir()
    return os.path.join(OUTPUT_DIR, filename)

def save_dataframe(df, filename, index=False):
    """Save DataFrame to CSV in output directory"""
    filepath = get_output_path(filename)
    df.to_csv(filepath, index=index)
    logger.info(f"Saved DataFrame to {filepath}")
    return filepath

def save_json(data, filename):
    """Save dictionary to JSON in output directory"""
    filepath = get_output_path(filename)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)
    logger.info(f"Saved JSON to {filepath}")
    return filepath

def save_plot(fig, filename):
    """Save matplotlib figure to output directory"""
    filepath = get_output_path(filename)
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    logger.info(f"Saved plot to {filepath}")
    plt.close(fig)
    return filepath

def calculate_metrics(y_true, y_pred):
    """Calculate MAE, RMSE, MAPE, Accuracy"""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
    import numpy as np
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred)
    accuracy = (1 - mape) * 100
    
    return {
        'mae': round(float(mae), 2),
        'rmse': round(float(rmse), 2),
        'mape': round(float(mape * 100), 2),
        'accuracy': round(float(accuracy), 2)
    }