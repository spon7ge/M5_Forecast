"""
Feature Engineering Pipeline for M5 Forecasting
Implements time window aggregations, lag features, rolling statistics using Spark DataFrames
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.window import Window
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import os
from datetime import datetime, timedelta
from config import get_config
import warnings
warnings.filterwarnings('ignore')

class M5FeatureEngineer:
    """Feature engineering pipeline for M5 forecasting using Spark DataFrames"""
    
    def __init__(self, app_name="M5-Feature-Engineering"):
        """Initialize Spark session with optimized configurations"""
        python_path = r"C:\Users\alexg\AppData\Local\Programs\Python\Python310\python.exe"
        os.environ['PYSPARK_PYTHON'] = python_path
        os.environ['PYSPARK_DRIVER_PYTHON'] = python_path
        
        self.spark = SparkSession.builder \
            .appName(app_name) \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
            .config("spark.driver.bindAddress", "127.0.0.1") \
            .config("spark.driver.host", "127.0.0.1") \
            .config("spark.ui.enabled", "false") \
            .config("spark.sql.adaptive.skewJoin.enabled", "true") \
            .config("spark.sql.adaptive.localShuffleReader.enabled", "true") \
            .config("spark.pyspark.python", python_path) \
            .config("spark.pyspark.driver.python", python_path) \
            .config("spark.python.worker.memory", "512m") \
            .config("spark.executor.memory", "1g") \
            .config("spark.driver.memory", "2g") \
            .config("spark.python.worker.reuse", "true") \
            .config("spark.executorEnv.PYSPARK_PYTHON", python_path) \
            .config("spark.executorEnv.PYSPARK_DRIVER_PYTHON", python_path) \
            .getOrCreate()
        
        self.spark.sparkContext.setLogLevel("WARN")
        self.config = get_config("development")
        
    def create_sample_sales_data(self, num_items=1000, num_days=365):
        """Create sample sales data with time series patterns for demonstration"""
        
        print(f"Creating sample sales data: {num_items} items x {num_days} days")
        
        # Generate realistic M5-style sales data
        data = []
        stores = ['CA_1', 'CA_2', 'CA_3', 'TX_1', 'TX_2', 'TX_3', 'WI_1', 'WI_2', 'WI_3']
        categories = ['HOBBIES', 'HOUSEHOLD', 'FOODS']
        departments = ['HOBBIES_1', 'HOUSEHOLD_1', 'FOODS_1', 'FOODS_2', 'FOODS_3']
        
        start_date = datetime(2011, 1, 29)
        
        for item_idx in range(num_items):
            store_id = str(np.random.choice(stores))
            state_id = store_id.split('_')[0]
            cat_id = str(np.random.choice(categories))
            dept_id = str(np.random.choice(departments))
            
            # Create base demand pattern with trends and seasonality
            base_demand = np.random.uniform(2, 15)  # Base daily demand
            trend = np.random.uniform(-0.002, 0.005)  # Daily trend
            
            for day_idx in range(num_days):
                current_date = start_date + timedelta(days=day_idx)
                
                # Add seasonality patterns
                day_of_week = current_date.weekday()
                month = current_date.month
                
                # Weekly seasonality (weekend boost for some categories)
                weekly_multiplier = 1.0
                if cat_id == 'FOODS' and day_of_week in [5, 6]:  # Weekend food boost
                    weekly_multiplier = 1.3
                elif cat_id == 'HOBBIES' and day_of_week in [4, 5, 6]:  # Weekend hobby boost
                    weekly_multiplier = 1.4
                    
                # Monthly seasonality (holiday effects)
                monthly_multiplier = 1.0
                if month in [11, 12]:  # Holiday season
                    monthly_multiplier = 1.5
                elif month in [6, 7, 8]:  # Summer season
                    monthly_multiplier = 1.2
                    
                # Calculate demand with trend, seasonality, and noise
                demand = base_demand * (1 + trend * day_idx) * weekly_multiplier * monthly_multiplier
                demand = np.maximum(0, int(demand + np.random.normal(0, demand * 0.3)))
                
                # Create 5-minute and 15-minute intervals within the day
                # Simulate intraday patterns
                for interval_5min in range(288):  # 24 hours * 12 (5-min intervals)
                    hour = interval_5min // 12
                    
                    # Intraday demand distribution (higher during business hours)
                    if 6 <= hour <= 22:  # Business hours
                        interval_demand = np.maximum(0, int(np.random.poisson(demand / 204)))  # 17 hours * 12 intervals
                    else:
                        interval_demand = np.maximum(0, int(np.random.poisson(demand / 1440)))  # Night hours
                    
                    timestamp = current_date + timedelta(minutes=interval_5min * 5)
                    
                    data.append({
                        'item_id': f'ITEM_{item_idx:04d}',
                        'store_id': store_id,
                        'state_id': state_id,
                        'cat_id': cat_id,
                        'dept_id': dept_id,
                        'timestamp': timestamp,
                        'date': current_date.date(),
                        'hour': hour,
                        'day_of_week': day_of_week,
                        'month': month,
                        'year': current_date.year,
                        'demand_5min': interval_demand,
                        'interval_5min': interval_5min
                    })
        
        # Convert to Spark DataFrame with explicit schema
        df_pandas = pd.DataFrame(data)
        
        # Ensure proper data types for Spark compatibility
        df_pandas['item_id'] = df_pandas['item_id'].astype(str)
        df_pandas['store_id'] = df_pandas['store_id'].astype(str)
        df_pandas['state_id'] = df_pandas['state_id'].astype(str)
        df_pandas['cat_id'] = df_pandas['cat_id'].astype(str)
        df_pandas['dept_id'] = df_pandas['dept_id'].astype(str)
        df_pandas['timestamp'] = pd.to_datetime(df_pandas['timestamp'])
        df_pandas['date'] = pd.to_datetime(df_pandas['date'])
        df_pandas['hour'] = df_pandas['hour'].astype(int)
        df_pandas['day_of_week'] = df_pandas['day_of_week'].astype(int)
        df_pandas['month'] = df_pandas['month'].astype(int)
        df_pandas['year'] = df_pandas['year'].astype(int)
        df_pandas['demand_5min'] = df_pandas['demand_5min'].astype(int)
        df_pandas['interval_5min'] = df_pandas['interval_5min'].astype(int)
        
        df_spark = self.spark.createDataFrame(df_pandas)
        
        print(f"Created sales data with {df_spark.count()} records")
        return df_spark
    
    def create_time_window_aggregations(self, df):
        """Create 5-minute and 15-minute time window aggregations"""
        
        print("Creating time window aggregations...")
        
        # Add 15-minute interval column (group every 3 intervals of 5 minutes)
        df_with_intervals = df.withColumn(
            "interval_15min", 
            (col("interval_5min") / 3).cast("int")
        ).withColumn(
            "timestamp_5min", 
            col("timestamp")
        ).withColumn(
            "timestamp_15min",
            from_unixtime(
                unix_timestamp(col("timestamp")) - 
                (unix_timestamp(col("timestamp")) % 900)  # Round down to 15 min
            ).cast("timestamp")
        )
        
        # 5-minute aggregations
        agg_5min = df_with_intervals.groupBy(
            "item_id", "store_id", "state_id", "cat_id", "dept_id",
            "date", "timestamp_5min", "hour", "day_of_week", "month", "year"
        ).agg(
            sum("demand_5min").alias("demand_5min_total"),
            avg("demand_5min").alias("demand_5min_avg"),
            count("demand_5min").alias("demand_5min_count"),
            stddev("demand_5min").alias("demand_5min_std")
        )
        
        # 15-minute aggregations  
        agg_15min = df_with_intervals.groupBy(
            "item_id", "store_id", "state_id", "cat_id", "dept_id", 
            "date", "timestamp_15min", "hour", "day_of_week", "month", "year"
        ).agg(
            sum("demand_5min").alias("demand_15min_total"),
            avg("demand_5min").alias("demand_15min_avg"),
            count("demand_5min").alias("demand_15min_count"),
            stddev("demand_5min").alias("demand_15min_std"),
            min("demand_5min").alias("demand_15min_min"),
            max("demand_5min").alias("demand_15min_max")
        )
        
        return agg_5min, agg_15min
    
    def create_lag_features(self, df, time_col="timestamp_15min", target_col="demand_15min_total", lag_periods=[1, 2, 3, 6, 12, 24]):
        """Create lag features using Spark window functions"""
        
        print(f"Creating lag features: {lag_periods}")
        
        # Define window specification for lag features
        window_spec = Window.partitionBy("item_id", "store_id").orderBy(time_col)
        
        # Add lag features
        df_with_lags = df
        for lag_value in lag_periods:
            df_with_lags = df_with_lags.withColumn(
                f"{target_col}_lag_{lag_value}",
                lag(col(target_col), lag_value).over(window_spec)
            )
        
        return df_with_lags
    
    def create_rolling_features(self, df, time_col="timestamp_15min", target_col="demand_15min_total", windows=[3, 6, 12, 24, 48]):
        """Create rolling averages and moving standard deviation features"""
        
        print(f"Creating rolling features with windows: {windows}")
        
        df_with_rolling = df
        
        for window_size in windows:
            # Define rolling window
            rolling_window = Window.partitionBy("item_id", "store_id") \
                                 .orderBy(time_col) \
                                 .rowsBetween(-window_size + 1, 0)
            
            # Rolling average
            df_with_rolling = df_with_rolling.withColumn(
                f"{target_col}_rolling_avg_{window_size}",
                avg(col(target_col)).over(rolling_window)
            )
            
            # Rolling standard deviation
            df_with_rolling = df_with_rolling.withColumn(
                f"{target_col}_rolling_std_{window_size}",
                stddev(col(target_col)).over(rolling_window)
            )
            
            # Rolling min/max
            df_with_rolling = df_with_rolling.withColumn(
                f"{target_col}_rolling_min_{window_size}",
                min(col(target_col)).over(rolling_window)
            )
            
            df_with_rolling = df_with_rolling.withColumn(
                f"{target_col}_rolling_max_{window_size}",
                max(col(target_col)).over(rolling_window)
            )
        
        return df_with_rolling
    
    def create_calendar_features(self, df):
        """Create additional calendar-based features"""
        
        print("Creating calendar features...")
        
        # Convert timestamp to date components
        df_calendar = df.withColumn("timestamp_dt", col("timestamp_15min")) \
                       .withColumn("hour_of_day", hour(col("timestamp_dt"))) \
                       .withColumn("day_of_month", dayofmonth(col("timestamp_dt"))) \
                       .withColumn("day_of_year", dayofyear(col("timestamp_dt"))) \
                       .withColumn("week_of_year", weekofyear(col("timestamp_dt"))) \
                       .withColumn("quarter", quarter(col("timestamp_dt"))) \
                       .withColumn("is_weekend", when(col("day_of_week").isin([5, 6]), 1).otherwise(0)) \
                       .withColumn("is_month_start", when(dayofmonth(col("timestamp_dt")) <= 3, 1).otherwise(0)) \
                       .withColumn("is_month_end", when(dayofmonth(col("timestamp_dt")) >= 28, 1).otherwise(0)) \
                       .withColumn("is_quarter_start", when(
                           (col("month").isin([1, 4, 7, 10]) & (dayofmonth(col("timestamp_dt")) <= 7)), 1).otherwise(0)) \
                       .withColumn("is_quarter_end", when(
                           (col("month").isin([3, 6, 9, 12]) & (dayofmonth(col("timestamp_dt")) >= 24)), 1).otherwise(0))
        
        return df_calendar
    
    def create_target_variable(self, df, target_col="demand_15min_total", horizon=1):
        """Create target variable for next time window prediction"""
        
        print(f"Creating target variable with horizon: {horizon}")
        
        # Define window for future values
        window_spec = Window.partitionBy("item_id", "store_id").orderBy("timestamp_15min")
        
        # Create future target
        df_target = df.withColumn(
            f"target_{target_col}_next_{horizon}",
            lead(col(target_col), horizon).over(window_spec)
        )
        
        return df_target
    
    def prepare_features_for_ml(self, df):
        """Prepare features for machine learning by selecting and cleaning"""
        
        print("Preparing features for ML...")
        
        # Select feature columns (exclude metadata and target)
        feature_cols = [col_name for col_name in df.columns if not any(x in col_name.lower() for x in 
                       ['timestamp', 'date', 'target_', 'item_id', 'store_id'])]
        
        target_col = [col_name for col_name in df.columns if col_name.startswith('target_')][0]
        
        # Select features and target, drop rows with null target
        ml_df = df.select(
            col("item_id"), col("store_id"), col("timestamp_15min"),
            *[col(c) for c in feature_cols],
            col(target_col)
        ).filter(col(target_col).isNotNull())
        
        # Fill null values with 0 for feature columns
        for col_name in feature_cols:
            ml_df = ml_df.fillna(0, subset=[col_name])
        
        return ml_df, feature_cols, target_col
    
    def train_lightgbm_model(self, df, feature_cols, target_col, test_size=0.2):
        """Train LightGBM regression model"""
        
        print("Training LightGBM model...")
        
        # Convert to Pandas for ML training
        # Convert categorical columns to numeric first
        categorical_cols = ['state_id', 'cat_id', 'dept_id']
        df_processed = df
        
        for col in categorical_cols:
            if col in feature_cols:
                from pyspark.ml.feature import StringIndexer
                indexer = StringIndexer(inputCol=col, outputCol=f"{col}_idx", handleInvalid="keep")
                df_processed = indexer.fit(df_processed).transform(df_processed)
                # Update feature columns list
                feature_cols[feature_cols.index(col)] = f"{col}_idx"
        ml_data = df_processed.toPandas()
        
        # Prepare features and target
        X = ml_data[feature_cols].fillna(0)
        y = ml_data[target_col]
        
        # Split data chronologically (last 20% for testing)
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        print(f"Training set size: {len(X_train)}")
        print(f"Test set size: {len(X_test)}")
        
        # LightGBM parameters
        lgb_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.1,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42
        }
        
        # Create datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
        
        # Train model
        model = lgb.train(
            params=lgb_params,
            train_set=train_data,
            valid_sets=[train_data, valid_data],
            valid_names=['train', 'valid'],
            num_boost_round=1000,
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)]
        )
        
        # Make predictions
        y_pred_train = model.predict(X_train, num_iteration=model.best_iteration)
        y_pred_test = model.predict(X_test, num_iteration=model.best_iteration)
        
        # Calculate metrics
        train_mae = mean_absolute_error(y_train, y_pred_train)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_mae = mean_absolute_error(y_test, y_pred_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        
        metrics = {
            'train_mae': train_mae,
            'train_rmse': train_rmse,
            'test_mae': test_mae,
            'test_rmse': test_rmse
        }
        
        return model, metrics, (X_test, y_test, y_pred_test)
    
    def save_model(self, model, metrics, model_path="models/", model_name=None):
        """Save trained model using joblib"""
        
        if model_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = f"m5_lightgbm_model_{timestamp}"
        
        os.makedirs(model_path, exist_ok=True)
        
        # Save model
        model_file = os.path.join(model_path, f"{model_name}.pkl")
        joblib.dump(model, model_file)
        
        # Save metrics
        metrics_file = os.path.join(model_path, f"{model_name}_metrics.pkl")
        joblib.dump(metrics, metrics_file)
        
        print(f"Model saved to: {model_file}")
        print(f"Metrics saved to: {metrics_file}")
        
        return model_file, metrics_file
    
    def run_complete_pipeline(self, num_items=500, num_days=90):
        """Run the complete feature engineering and model training pipeline"""
        
        print("=== Starting M5 Feature Engineering & Model Training Pipeline ===")
        
        try:
            # Step 1: Create sample data
            sales_df = self.create_sample_sales_data(num_items, num_days)
            
            # Step 2: Create time window aggregations
            agg_5min, agg_15min = self.create_time_window_aggregations(sales_df)
            
            # Step 3: Focus on 15-minute aggregations for modeling
            print("Processing 15-minute aggregations...")
            
            # Step 4: Create lag features
            df_with_lags = self.create_lag_features(agg_15min)
            
            # Step 5: Create rolling features
            df_with_rolling = self.create_rolling_features(df_with_lags)
            
            # Step 6: Create calendar features
            df_with_calendar = self.create_calendar_features(df_with_rolling)
            
            # Step 7: Create target variable
            df_with_target = self.create_target_variable(df_with_calendar)
            
            # Step 8: Prepare for ML
            ml_df, feature_cols, target_col = self.prepare_features_for_ml(df_with_target)
            
            print(f"Feature columns ({len(feature_cols)}): {feature_cols[:10]}...")
            print(f"Target column: {target_col}")
            print(f"ML dataset size: {ml_df.count()} records")
            
            # Step 9: Train model
            model, metrics, test_results = self.train_lightgbm_model(ml_df, feature_cols, target_col)
            
            # Step 10: Print results
            print("\n=== Model Performance ===")
            for metric, value in metrics.items():
                print(f"{metric}: {value:.4f}")
            
            # Step 11: Save model
            model_files = self.save_model(model, metrics)
            
            print("\n=== Pipeline Completed Successfully ===")
            return model, metrics, model_files
            
        except Exception as e:
            print(f"Pipeline failed with error: {str(e)}")
            raise e
        finally:
            self.spark.stop()

def main():
    """Main execution function"""
    
    # Initialize feature engineer
    engineer = M5FeatureEngineer()
    
    # Run complete pipeline
    model, metrics, model_files = engineer.run_complete_pipeline(
        num_items=300,  # Reduced for faster execution
        num_days=60     # 2 months of data
    )
    
    print(f"\nModel training completed!")
    print(f"Model files: {model_files}")

if __name__ == "__main__":
    main()