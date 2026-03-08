import sys
import os

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from forecasting_pipeline import ForecastingPipeline

def main():
    """
    Main execution function
    """
    print("=" * 60)
    print("STARTING SALES FORECASTING SYSTEM")
    print("=" * 60)
    
    try:
        # Initialize pipeline
        pipeline = ForecastingPipeline()
        
        # Run complete analysis with output saving enabled
        results = pipeline.run_complete_analysis(save_outputs=True)
        
        print("\nSUCCESS! Analysis Complete!")
        print("\nGenerated Files in 'output/' folder:")
        print("   - sales_forecast.png - Forecast visualization")
        print("   - forecast_results.csv - 30-day forecast")
        print("   - revenue_opportunities.csv - Optimization opportunities")
        print("   - feature_importance.csv - ML feature rankings")
        print("   - model_metrics.json - Performance metrics")
        
        return results
        
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = main()