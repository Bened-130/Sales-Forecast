import sys
import os

# Add src directory to path if needed
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from forecasting_pipeline import ForecastingPipeline

def main():
    """
    Main execution function
    Run the complete sales forecasting pipeline
    """
    print("🚀 Starting Sales Forecasting System")
    print("=" * 60)
    
    try:
        # Initialize pipeline
        pipeline = ForecastingPipeline()
        
        # Run complete analysis with output saving enabled
        results = pipeline.run_complete_analysis(save_outputs=True)
        
        print("\n✅ Analysis Complete!")
        print("\n📁 Generated Files:")
        print("   • output/forecast_results.csv - 30-day forecast")
        print("   • output/revenue_opportunities.csv - Optimization opportunities")
        print("   • output/feature_importance.csv - ML feature rankings")
        print("   • output/model_metrics.json - Performance metrics")
        print("   • output/sales_forecast.png - Forecast visualization")
        
        return results
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = main()