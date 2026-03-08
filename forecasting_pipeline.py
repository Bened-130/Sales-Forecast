    def _print_summary(self, prophet_metrics, ml_metrics, opportunities, feature_importance):
        """Print results summary"""
        print(f"\n{'='*60}")
        print("FORECASTING RESULTS SUMMARY")
        print(f"{'='*60}")
        
        print(f"\nProphet Model Performance:")
        print(f"   Accuracy: {prophet_metrics['accuracy']}%")
        print(f"   MAE: ${prophet_metrics['mae']:,.2f}")
        print(f"   RMSE: ${prophet_metrics['rmse']:,.2f}")
        
        print(f"\nML Model (Random Forest) Performance:")
        print(f"   Accuracy: {ml_metrics['accuracy']}%")
        print(f"   MAE: ${ml_metrics['mae']:,.2f}")
        print(f"   RMSE: ${ml_metrics['rmse']:,.2f}")
        
        print(f"\nRevenue Opportunities:")
        total_opp = opportunities['opportunity_value'].sum()
        print(f"   Total Opportunity: ${total_opp:,.2f}")
        
        print(f"\nTop 5 Important Features:")
        if feature_importance is not None:
            print(feature_importance.head().to_string(index=False))
        
        print(f"\n{'='*60}")