#!/usr/bin/env python3
"""
Standalone script to run M5 Feature Engineering and Model Training Pipeline
Run this script to execute the complete pipeline with sample data.
"""

import os
import sys
from datetime import datetime

def main():
    """Run the complete M5 feature engineering pipeline"""
    
    print("=" * 60)
    print("M5 FEATURE ENGINEERING & BATCH MODEL TRAINING")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        # Import the feature engineering module
        from featureEngineering import M5FeatureEngineer
        
        print("✓ Successfully imported featureEngineering module")
        
        # Initialize the pipeline
        engineer = M5FeatureEngineer("M5-Production-Pipeline")
        print("✓ Initialized Spark session")
        
        # Run the complete pipeline with production-sized data
        print("\nStarting feature engineering pipeline...")
        print("This may take several minutes...")
        
        model, metrics, model_files = engineer.run_complete_pipeline(
            num_items=200,  # Moderate size for demo
            num_days=45     # 1.5 months of data
        )
        
        print("\n" + "=" * 60)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        print("\nFinal Model Performance:")
        print("-" * 30)
        for metric, value in metrics.items():
            print(f"{metric:15s}: {value:8.4f}")
        
        print(f"\nModel Files:")
        print(f"- Model: {model_files[0]}")
        print(f"- Metrics: {model_files[1]}")
        
        print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("\nPlease install required packages:")
        print("pip install -r requirements_ml.txt")
        return False
        
    except Exception as e:
        print(f"❌ Pipeline Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)