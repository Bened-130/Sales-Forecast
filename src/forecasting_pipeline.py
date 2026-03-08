"""
Forecasting Pipeline Module
Complete end-to-end sales forecasting pipeline
"""

import pandas as pd
import json
from config import logger, FORECAST_PERIODS, TRAIN_TEST_SPLIT
from sales_data_processor import SalesDataProcessor
from prophet_forecaster import ProphetForecaster
from ml_forecaster import MLForecaster
from revenue_optimizer import RevenueOptimizer
from utils import save_json, save_dataframe

class ForecastingPipeline:
    """Complete forecasting pipeline"""
    
    def __init__(self):
        self.processor = SalesDataProcessor()
        self.prophet_model = ProphetForecaster()
        self.ml_model = MLForecaster('random_forest')
        self.optimizer = RevenueOptimizer()
        self.results = {}
        logger.info("ForecastingPipeline initialized")
    
    def run_complete_analysis(self, save_outputs=True):
        """Run complete forecasting analysis"""
        logger.info("=" * 60)
        logger.info("STARTING SALES FORECASTING PIPELINE")
        logger.info("=" * 60)
        
        # Step 1: Generate/Load Data
        logger.info("\n--- STEP 1: Data Processing ---")
        sales_data = self.processor.generate_sample_data()
        sales_clean = self.processor.clean_data(sales_data)
        
        # Step 2: Prepare data for Prophet
        logger.info("\n--- STEP 2: Prophet Forecasting ---")
        daily_sales = self.processor.aggregate_daily_sales(sales_clean)
        
        # Split data
        train_size = int(len(daily_sales) * TRAIN_TEST_SPLIT)
        train_data = daily_sales[:train_size]
        test_data = daily_sales[train_size:]
        
        # Train Prophet
        self.prophet_model.train(train_data)
        prophet_metrics = self.prophet_model.evaluate(test_data)
        
        # Generate forecast
        forecast = self.prophet_model.predict(periods=FORECAST_PERIODS)
        
        # Step 3: ML Model
        logger.info("\n--- STEP 3: ML Forecasting ---")
        
        # Prepare features
        ml_data = sales_clean.groupby('date').agg({
            'revenue': 'sum'
        }).reset_index()
        
        ml_features = self.ml_model.create_features(ml_data)
        
        # Split features
        X_train, X_test, y_train, y_test = self.ml_model.prepare_train_test_split(ml_features)
        
        # Train ML model
        self.ml_model.train(X_train, y_train)
        ml_metrics = self.ml_model.evaluate(X_test, y_test)
        
        # Feature importance
        feature_importance = self.ml_model.feature_importance()
        
        # Step 4: Revenue Optimization
        logger.info("\n--- STEP 4: Revenue Optimization ---")
        opportunities = self.optimizer.identify_opportunities(sales_clean)
        optimized_inventory = self.optimizer.optimize_inventory(forecast)
        regional_stats = self.optimizer.regional_analysis(sales_clean)
        product_stats = self.optimizer.product_analysis(sales_clean)
        
        # Step 5: Compile Results
        logger.info("\n" + "=" * 60)
        logger.info("FORECASTING PIPELINE RESULTS")
        logger.info("=" * 60)
        
        self.results = {
            'prophet_metrics': prophet_metrics,
            'ml_metrics': ml_metrics,
            'forecast': forecast,
            'opportunities': opportunities,
            'feature_importance': feature_importance,
            'optimized_inventory': optimized_inventory,
            'regional_stats': regional_stats,
            'product_stats': product_stats
        }
        
        # Print summary
        self._print_summary(prophet_metrics, ml_metrics, opportunities, feature_importance)
        
        # Save outputs if requested
        if save_outputs:
            self._save_all_outputs()
        
        logger.info("\n" + "=" * 60)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        
        return self.results
    
    def _print_summary(self, prophet_metrics, ml_metrics, opportunities, feature_importance):
        """Print results summary"""
        print(f"\n{'='*60}")
        print("FORECASTING RESULTS SUMMARY")
        print(f"{'='*60}")
        
        print(f"\n📊 Prophet Model Performance:")
        print(f"   Accuracy: {prophet_metrics['accuracy']}%")
        print(f"   MAE: ${prophet_metrics['mae']:,.2f}")
        print(f"   RMSE: ${prophet_metrics['rmse']:,.2f}")
        print(f"   MAPE: {prophet_metrics['mape']}%")
        
        print(f"\n🤖 ML Model (Random Forest) Performance:")
        print(f"   Accuracy: {ml_metrics['accuracy']}%")
        print(f"   MAE: ${ml_metrics['mae']:,.2f}")
        print(f"   RMSE: ${ml_metrics['rmse']:,.2f}")
        print(f"   MAPE: {ml_metrics['mape']}%")
        
        print(f"\n💰 Revenue Opportunities:")
        total_opp = opportunities['opportunity_value'].sum()
        print(f"   Total Opportunity: ${total_opp:,.2f}")
        print(f"   Number of Opportunities: {len(opportunities[opportunities['opportunity_value'] > 0])}")
        
        print(f"\n🏆 Top 5 Revenue Opportunities:")
        top_opp = opportunities[opportunities['opportunity_value'] > 0].head()
        if not top_opp.empty:
            print(top_opp[['region', 'product', 'opportunity_value', 'improvement_pct']].to_string(index=False))
        else:
            print("   No significant opportunities identified")
        
        print(f"\n🔍 Top 5 Important Features:")
        if feature_importance is not None:
            print(feature_importance.head().to_string(index=False))
        
        print(f"\n{'='*60}")
    
    def _save_all_outputs(self):
        """Save all outputs to files"""
        from config import (
            FORECAST_CSV, OPPORTUNITIES_CSV, FEATURE_IMPORTANCE_CSV, 
            METRICS_JSON, FORECAST_PLOT
        )
        
        logger.info("\n--- Saving Outputs ---")
        
        # Save forecast
        if self.results.get('forecast') is not None:
            self.prophet_model.save_forecast(FORECAST_CSV)
        
        # Save opportunities
        if self.results.get('opportunities') is not None:
            self.optimizer.save_opportunities(self.results['opportunities'], OPPORTUNITIES_CSV)
        
        # Save feature importance
        self.ml_model.save_feature_importance(FEATURE_IMPORTANCE_CSV)
        
        # Save metrics
        metrics = {
            'prophet': self.results['prophet_metrics'],
            'ml_model': self.results['ml_metrics'],
            'timestamp': pd.Timestamp.now().isoformat()
        }
        save_json(metrics, METRICS_JSON)
        
        # Save plots
        self.prophet_model.plot_forecast(FORECAST_PLOT)
        
        logger.info("All outputs saved successfully")
    
    def get_results(self):
        """Get all results"""
        return self.results
    
    def get_forecast_plot(self):
        """Get forecast plot"""
        return self.prophet_model.plot_forecast()