"""
Quick test to verify all modules work
"""

import warnings
warnings.filterwarnings('ignore')

print("Testing imports...")
try:
    from config import logger, OUTPUT_DIR
    print("✅ config.py loaded")
    
    from utils import ensure_output_dir
    print("✅ utils.py loaded")
    
    from sales_data_processor import SalesDataProcessor
    print("✅ sales_data_processor.py loaded")
    
    from prophet_forecaster import ProphetForecaster
    print("✅ prophet_forecaster.py loaded")
    
    from ml_forecaster import MLForecaster
    print("✅ ml_forecaster.py loaded")
    
    from revenue_optimizer import RevenueOptimizer
    print("✅ revenue_optimizer.py loaded")
    
    from forecasting_pipeline import ForecastingPipeline
    print("✅ forecasting_pipeline.py loaded")
    
    print("\n✅ All modules imported successfully!")
    print(f"📁 Output directory: {OUTPUT_DIR}")
    
    # Quick run
    print("\n🚀 Running quick test...")
    pipeline = ForecastingPipeline()
    results = pipeline.run_complete_analysis(save_outputs=True)
    print("\n✅ Test completed successfully!")
    
except Exception as e:
    print(f"\n Error: {e}")
    import traceback
    traceback.print_exc()