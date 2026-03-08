import pandas as pd
import numpy as np
from config import logger, STOCK_BUFFER, SAFETY_STOCK, REORDER_POINT
from utils import save_dataframe

class RevenueOptimizer:
    """Identify revenue opportunities"""
    
    def __init__(self):
        logger.info("RevenueOptimizer initialized")
    
    def identify_opportunities(self, df):
        """Identify revenue optimization opportunities"""
        logger.info("Identifying revenue opportunities")
        
        # Calculate product benchmarks
        product_benchmark = df.groupby('product')['revenue'].agg([
            ('avg_revenue', 'mean'),
            ('total_revenue', 'sum')
        ]).reset_index()
        
        # Calculate regional performance
        regional_performance = df.groupby(['region', 'product'])['revenue'].agg([
            ('actual_revenue', 'sum'),
            ('avg_daily_revenue', 'mean'),
            ('transaction_count', 'count')
        ]).reset_index()
        
        # Merge with benchmarks
        opportunities = regional_performance.merge(
            product_benchmark[['product', 'avg_revenue']], 
            on='product', 
            suffixes=('', '_benchmark')
        )
        
        # Calculate gap
        opportunities['expected_revenue'] = opportunities['avg_revenue'] * opportunities['transaction_count']
        opportunities['revenue_gap'] = opportunities['expected_revenue'] - opportunities['actual_revenue']
        opportunities['opportunity_value'] = opportunities['revenue_gap'].clip(lower=0)
        opportunities['improvement_pct'] = (opportunities['opportunity_value'] / opportunities['actual_revenue'] * 100).round(2)
        
        total_opportunity = opportunities['opportunity_value'].sum()
        logger.info(f"Total revenue opportunity identified: ${total_opportunity:,.2f}")
        
        return opportunities.sort_values('opportunity_value', ascending=False)
    
    def optimize_inventory(self, forecast_df):
        """Optimize inventory based on forecast"""
        logger.info("Optimizing inventory levels")
        
        result_df = forecast_df.copy()
        
        result_df['recommended_stock'] = (result_df['yhat'] * STOCK_BUFFER).round(2)
        result_df['safety_stock'] = (result_df['yhat'] * SAFETY_STOCK).round(2)
        result_df['reorder_point'] = (result_df['yhat'] * REORDER_POINT).round(2)
        result_df['max_stock'] = (result_df['yhat'] * (STOCK_BUFFER + SAFETY_STOCK)).round(2)
        
        return result_df
    
    def regional_analysis(self, df):
        """Analyze performance by region"""
        regional_stats = df.groupby('region').agg({
            'revenue': ['sum', 'mean', 'count'],
            'quantity_sold': 'sum'
        }).round(2)
        
        regional_stats.columns = ['total_revenue', 'avg_revenue', 'transaction_count', 'total_quantity']
        regional_stats = regional_stats.reset_index()
        regional_stats['revenue_per_transaction'] = (regional_stats['total_revenue'] / regional_stats['transaction_count']).round(2)
        
        return regional_stats.sort_values('total_revenue', ascending=False)
    
    def product_analysis(self, df):
        """Analyze performance by product"""
        product_stats = df.groupby('product').agg({
            'revenue': ['sum', 'mean', 'count'],
            'quantity_sold': 'sum',
            'unit_price': 'mean'
        }).round(2)
        
        product_stats.columns = ['total_revenue', 'avg_revenue', 'transaction_count', 'total_quantity', 'avg_price']
        product_stats = product_stats.reset_index()
        product_stats['market_share_pct'] = (product_stats['total_revenue'] / product_stats['total_revenue'].sum() * 100).round(2)
        
        return product_stats.sort_values('total_revenue', ascending=False)
    
    def save_opportunities(self, opportunities, filename):
        """Save opportunities to CSV"""
        save_dataframe(opportunities, filename)
        return True