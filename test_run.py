"""
Quick test to verify all modules work
Place this file in the ROOT folder (same level as main.py)
"""

import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("SALES FORECASTING SYSTEM - QUICK TEST")
print("="*60)

# Step 1: Check dependencies
print("\n📦 Step 1: Checking Dependencies...")
required_packages = ['pandas', 'numpy', 'sklearn', 'prophet', 'matplotlib', 'scipy']

missing = []
for package in required_packages:
    try:
        if package == 'sklearn':
            import sklearn
            print(f"   ✅ scikit-learn")
        else:
            __import__(package)
            print(f"   ✅ {package}")
    except ImportError:
        print(f"   ❌ {package} - MISSING")
        missing.append(package)

if missing:
    print(f"\n❌ Missing packages: {', '.join(missing)}")
    print("💡 Install with: pip install -r requirements.txt")
    exit(1)

# Step 2: Test imports
print("\n📂 Step 2: Testing Module Imports...")

try:
    from config import logger, OUTPUT_DIR
    print("   ✅ config.py")
    
    from utils import ensure_output_dir
    print("   ✅ utils.py")
    
    from sales_data_processor import SalesDataProcessor
    print("   ✅ sales_data_processor.py")
    
    from prophet_forecaster import ProphetForecaster
    print("   ✅ prophet_forecaster.py")
    
    from ml_forecaster import MLForecaster
    print("   ✅ ml_forecaster.py")
    
    from revenue_optimizer import RevenueOptimizer
    print("   ✅ revenue_optimizer.py")
    
    from forecasting_pipeline import ForecastingPipeline
    print("   ✅ forecasting_pipeline.py")
    
except Exception as e:
    print(f"\n❌ Import Error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Step 3: Run pipeline
print("\n🚀 Step 3: Running Forecasting Pipeline...")
print("-"*60)

try:
    pipeline = ForecastingPipeline()
    results = pipeline.run_complete_analysis(save_outputs=True)
    
    print("\n" + "="*60)
    print("✅ SUCCESS! All outputs generated in 'output/' folder:")
    print("="*60)
    print("   📊 sales_forecast.png")
    print("   📄 forecast_results.csv")
    print("   📄 revenue_opportunities.csv")
    print("   📄 feature_importance.csv")
    print("   📄 model_metrics.json")
    
except Exception as e:
    print(f"\n❌ Pipeline Error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n🎉 Test completed successfully!")