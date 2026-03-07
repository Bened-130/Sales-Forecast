import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from config import logger, DATA_PARAMS
from utils import save_dataframe

class SalesDataProcessor:
    """Process and clean sales data"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        logger.info("SalesDataProcessor initialized")
    
    def generate_sample_data(self, n_records=None):
        """
        Generate sample sales data for demonstration
        If n_records is None, generates data for all date combinations
        """
        logger.info(f"Generating sample sales records")
        
        np.random.seed(42)
        
        # Date range
        dates = pd.date_range(
            start=DATA_PARAMS['start_date'], 
            end=DATA_PARAMS['end_date'], 
            freq='D'
        )
        
        products = DATA_PARAMS['products']
        regions = DATA_PARAMS['regions']
        
        data = []
        record_count = 0
        
        for date in dates:
            for product in products:
                for region in regions:
                    if n_records and record_count >= n_records:
                        break
                    
                    # Base sales with seasonality
                    base_sales = np.random.randint(
                        DATA_PARAMS['base_quantity_min'], 
                        DATA_PARAMS['base_quantity_max']
                    )
                    
                    # Add seasonality (higher in hot months)
                    month = date.month
                    seasonal_factor = 1.0 + (0.3 if month in [1, 2, 12] else 0)
                    
                    # Add day of week effect
                    dow_factor = 1.2 if date.dayofweek in [5, 6] else 1.0
                    
                    # Calculate final quantity
                    quantity = int(base_sales * seasonal_factor * dow_factor)
                    
                    # Calculate revenue
                    unit_price = np.random.uniform(
                        DATA_PARAMS['unit_price_min'], 
                        DATA_PARAMS['unit_price_max']
                    )
                    revenue = quantity * unit_price
                    
                    data.append({
                        'date': date,
                        'product': product,
                        'region': region,
                        'quantity_sold': quantity,
                        'unit_price': round(unit_price, 2),
                        'revenue': round(revenue, 2),
                        'distribution_point_id': np.random.randint(1, DATA_PARAMS['n_distribution_points'] + 1)
                    })
                    record_count += 1
                
                if n_records and record_count >= n_records:
                    break
            
            if n_records and record_count >= n_records:
                break
        
        df = pd.DataFrame(data)
        logger.info(f"Generated {len(df)} sales records")
        return df
    
    def clean_data(self, df):
        """Clean and prepare sales data"""
        logger.info("Cleaning sales data")
        
        df = df.copy()
        
        # Remove duplicates
        initial_count = len(df)
        df = df.drop_duplicates()
        logger.info(f"Removed {initial_count - len(df)} duplicates")
        
        # Handle missing values
        df['quantity_sold'] = df['quantity_sold'].fillna(df['quantity_sold'].median())
        df['revenue'] = df['revenue'].fillna(df['revenue'].median())
        df['unit_price'] = df['unit_price'].fillna(df['unit_price'].median())
        
        # Remove outliers using IQR method
        for col in ['quantity_sold', 'revenue']:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        
        # Feature engineering
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        df['day_of_week'] = df['date'].dt.dayofweek
        df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        logger.info(f"Data cleaned: {len(df)} records remaining")
        return df
    
    def aggregate_daily_sales(self, df):
        """Aggregate sales by day for time series forecasting"""
        daily_sales = df.groupby('date').agg({
            'quantity_sold': 'sum',
            'revenue': 'sum'
        }).reset_index()
        
        daily_sales.columns = ['ds', 'quantity', 'y']
        return daily_sales
    
    def save_processed_data(self, df, filename='processed_sales_data.csv'):
        """Save processed data to CSV"""
        save_dataframe(df, filename)
        return filename