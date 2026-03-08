import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("SALES FORECASTING SYSTEM - QUICK TEST")
print("=" * 60)

# Step 1: Check dependencies
print("\nStep 1: Checking Dependencies...")
required_packages = ['pandas', 'numpy', 'sklearn', 'prophet', 'matplotlib', 'scipy']

missing = []
for package in required_packages:
    try:
        if package == 'sklearn':
            import sklearn
            print("   [OK] scikit-learn")
        else:
            __import__(package)
            print(f"   [OK] {package}")
    except ImportError:
        print(f"   [MISSING] {package}")
        missing.append(package)

if missing:
    print(f"\nMISSING PACKAGES: {', '.join(missing)}")
    print("Install with: pip install -r requirements.txt")
    exit(1)

# Step 2: Test imports
print("\nStep 2: Testing Module Imports...")

try:
    from config import logger, OUTPUT_DIR
    print("   [OK] config.py")
    
    from utils import ensure_output_dir
    print("   [OK] utils.py")
    
    from sales_data_processor import SalesDataProcessor
    print("   [OK] sales_data_processor.py")
    
    from prophet_forecaster import ProphetForecaster
    print("   [OK] prophet_forecaster.py")
    
    from ml_forecaster import MLForecaster
    print("   [OK] ml_forecaster.py")
    
    from revenue_optimizer import RevenueOptimizer
    print("   [OK] revenue_optimizer.py")
    
    from forecasting_pipeline import ForecastingPipeline
    print("   [OK] forecasting_pipeline.py")
    
except Exception as e:
    print(f"\nIMPORT ERROR: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Step 3: Run pipeline
print("\nStep 3: Running Forecasting Pipeline...")
print("-" * 60)

try:
    pipeline = ForecastingPipeline()
    results = pipeline.run_complete_analysis(save_outputs=True)
    
    print("\n" + "=" * 60)
    print("SUCCESS! All outputs generated in 'output/' folder:")
    print("=" * 60)
    print("   - sales_forecast.png")
    print("   - forecast_results.csv")
    print("   - revenue_opportunities.csv")
    print("   - feature_importance.csv")
    print("   - model_metrics.json")
    
except Exception as e:
    print(f"\nPIPELINE ERROR: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\nTest completed successfully!")