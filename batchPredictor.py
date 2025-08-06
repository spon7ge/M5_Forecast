"""
Batch Prediction Pipeline for M5 Forecasting
Uses trained LightGBM model to make predictions on new data
"""

import joblib
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class M5BatchPredictor:
    """Batch prediction pipeline for trained M5 forecasting models"""
    
    def __init__(self, model_path: str):
        """Initialize predictor with trained model"""
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.model = joblib.load(model_path)
        
        # Try to load metrics if available
        metrics_path = model_path.replace('.pkl', '_metrics.pkl')
        if os.path.exists(metrics_path):
            self.metrics = joblib.load(metrics_path)
        else:
            self.metrics = None
            
        print(f"✓ Loaded model from: {model_path}")
        if self.metrics:
            print(f"✓ Model performance metrics available")
            for metric, value in self.metrics.items():
                print(f"  {metric}: {value:.4f}")
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for prediction (simulate feature engineering)"""
        
        print("Preparing features for prediction...")
        
        # Ensure required columns exist
        required_base_cols = ['timestamp', 'item_id', 'store_id', 'demand']
        for col in required_base_cols:
            if col not in df.columns:
                raise ValueError(f"Required column missing: {col}")
        
        # Sort by timestamp for proper lag calculations
        df = df.sort_values(['item_id', 'store_id', 'timestamp']).reset_index(drop=True)
        
        # Convert timestamp if needed
        if df['timestamp'].dtype == 'object':
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Create time-based features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        df['year'] = df['timestamp'].dt.year
        df['day_of_month'] = df['timestamp'].dt.day
        df['day_of_year'] = df['timestamp'].dt.dayofyear
        df['week_of_year'] = df['timestamp'].dt.isocalendar().week
        df['quarter'] = df['timestamp'].dt.quarter
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_month_start'] = (df['day_of_month'] <= 3).astype(int)
        df['is_month_end'] = (df['day_of_month'] >= 28).astype(int)
        df['is_quarter_start'] = ((df['month'].isin([1, 4, 7, 10])) & (df['day_of_month'] <= 7)).astype(int)
        df['is_quarter_end'] = ((df['month'].isin([3, 6, 9, 12])) & (df['day_of_month'] >= 24)).astype(int)
        
        # Create lag features
        lag_windows = [1, 2, 3, 6, 12, 24]
        for lag in lag_windows:
            df[f'demand_15min_total_lag_{lag}'] = df.groupby(['item_id', 'store_id'])['demand'].shift(lag)
        
        # Create rolling features
        rolling_windows = [3, 6, 12, 24, 48]
        for window in rolling_windows:
            df[f'demand_15min_total_rolling_avg_{window}'] = (
                df.groupby(['item_id', 'store_id'])['demand']
                .rolling(window=window, min_periods=1)
                .mean()
                .reset_index(level=[0, 1], drop=True)
            )
            
            df[f'demand_15min_total_rolling_std_{window}'] = (
                df.groupby(['item_id', 'store_id'])['demand']
                .rolling(window=window, min_periods=1)
                .std()
                .reset_index(level=[0, 1], drop=True)
            )
            
            df[f'demand_15min_total_rolling_min_{window}'] = (
                df.groupby(['item_id', 'store_id'])['demand']
                .rolling(window=window, min_periods=1)
                .min()
                .reset_index(level=[0, 1], drop=True)
            )
            
            df[f'demand_15min_total_rolling_max_{window}'] = (
                df.groupby(['item_id', 'store_id'])['demand']
                .rolling(window=window, min_periods=1)
                .max()
                .reset_index(level=[0, 1], drop=True)
            )
        
        # Additional aggregated features
        df['demand_15min_avg'] = df['demand']  # Assuming this is the aggregated value
        df['demand_15min_count'] = 1  # Single observation per record
        df['demand_15min_std'] = 0    # Will be updated with actual std if available
        df['demand_15min_min'] = df['demand']
        df['demand_15min_max'] = df['demand']
        
        # Add categorical features (dummy encoding would be better for real scenarios)
        df['state_id_CA'] = (df.get('state_id', 'CA') == 'CA').astype(int)
        df['state_id_TX'] = (df.get('state_id', 'TX') == 'TX').astype(int)
        df['state_id_WI'] = (df.get('state_id', 'WI') == 'WI').astype(int)
        
        # Fill missing values
        feature_cols = [col for col in df.columns if col not in ['timestamp', 'item_id', 'store_id', 'demand']]
        df[feature_cols] = df[feature_cols].fillna(0)
        
        return df
    
    def predict(self, df: pd.DataFrame, feature_cols: Optional[List[str]] = None) -> np.ndarray:
        """Make predictions on prepared dataframe"""
        
        if feature_cols is None:
            # Use all columns except metadata
            feature_cols = [col for col in df.columns if col not in 
                          ['timestamp', 'item_id', 'store_id', 'demand', 'target_demand_15min_total_next_1']]
        
        # Select only available features
        available_features = [col for col in feature_cols if col in df.columns]
        
        if len(available_features) == 0:
            raise ValueError("No valid feature columns found for prediction")
        
        print(f"Using {len(available_features)} features for prediction")
        
        # Make predictions
        X = df[available_features].fillna(0)
        predictions = self.model.predict(X)
        
        return predictions
    
    def predict_batch(self, input_data: pd.DataFrame) -> pd.DataFrame:
        """Run complete batch prediction pipeline"""
        
        print(f"Starting batch prediction for {len(input_data)} records")
        
        # Prepare features
        df_features = self.prepare_features(input_data.copy())
        
        # Make predictions
        predictions = self.predict(df_features)
        
        # Create results dataframe
        results = pd.DataFrame({
            'item_id': df_features['item_id'],
            'store_id': df_features['store_id'],
            'timestamp': df_features['timestamp'],
            'actual_demand': df_features['demand'],
            'predicted_demand': predictions,
            'prediction_error': df_features['demand'] - predictions,
            'absolute_error': np.abs(df_features['demand'] - predictions)
        })
        
        return results
    
    def evaluate_predictions(self, results: pd.DataFrame) -> Dict[str, float]:
        """Evaluate prediction performance"""
        
        actual = results['actual_demand']
        predicted = results['predicted_demand']
        
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        
        mae = mean_absolute_error(actual, predicted)
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        mape = np.mean(np.abs((actual - predicted) / np.maximum(actual, 1e-6))) * 100
        
        metrics = {
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'mean_actual': actual.mean(),
            'mean_predicted': predicted.mean()
        }
        
        return metrics

def create_sample_prediction_data(num_records: int = 1000) -> pd.DataFrame:
    """Create sample data for prediction demonstration"""
    
    np.random.seed(42)
    
    # Generate timestamps (15-minute intervals)
    start_time = datetime(2024, 1, 1, 0, 0)
    timestamps = [start_time + timedelta(minutes=15*i) for i in range(num_records)]
    
    # Generate sample items and stores
    items = [f'ITEM_{i:04d}' for i in range(10)]
    stores = ['CA_1', 'CA_2', 'TX_1', 'TX_2', 'WI_1']
    
    data = []
    for i, timestamp in enumerate(timestamps):
        item_id = np.random.choice(items)
        store_id = np.random.choice(stores)
        
        # Generate demand with time-based patterns
        hour = timestamp.hour
        day_of_week = timestamp.weekday()
        
        # Base demand with hourly and weekly patterns
        base_demand = 5 + 10 * np.sin(2 * np.pi * hour / 24)  # Daily cycle
        if day_of_week >= 5:  # Weekend boost
            base_demand *= 1.3
            
        demand = np.maximum(0, int(base_demand + np.random.normal(0, 2)))
        
        data.append({
            'timestamp': timestamp,
            'item_id': item_id,
            'store_id': store_id,
            'state_id': store_id.split('_')[0],
            'demand': demand
        })
    
    return pd.DataFrame(data)

def main():
    """Demonstration of batch prediction pipeline"""
    
    print("=" * 60)
    print("M5 BATCH PREDICTION DEMO")
    print("=" * 60)
    
    # Find the most recent model
    model_dir = "models"
    if not os.path.exists(model_dir):
        print("❌ No models directory found. Please run feature engineering first.")
        return
    
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.pkl') and 'metrics' not in f]
    if not model_files:
        print("❌ No model files found. Please run feature engineering first.")
        return
    
    # Use the most recent model
    model_file = sorted(model_files)[-1]
    model_path = os.path.join(model_dir, model_file)
    
    try:
        # Initialize predictor
        predictor = M5BatchPredictor(model_path)
        
        # Create sample data
        print("\nCreating sample prediction data...")
        sample_data = create_sample_prediction_data(500)
        print(f"✓ Created {len(sample_data)} sample records")
        
        # Run batch prediction
        print("\nRunning batch predictions...")
        results = predictor.predict_batch(sample_data)
        
        # Evaluate results
        print("\nEvaluating predictions...")
        eval_metrics = predictor.evaluate_predictions(results)
        
        print("\nPrediction Results:")
        print("-" * 30)
        for metric, value in eval_metrics.items():
            print(f"{metric:15s}: {value:8.4f}")
        
        # Show sample results
        print("\nSample Predictions:")
        print("-" * 50)
        sample_results = results.head(10)
        for idx, row in sample_results.iterrows():
            print(f"{row['item_id']} @ {row['store_id']}: "
                  f"Actual={row['actual_demand']:5.1f}, "
                  f"Predicted={row['predicted_demand']:5.1f}, "
                  f"Error={row['prediction_error']:5.1f}")
        
        print(f"\n✓ Batch prediction completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()