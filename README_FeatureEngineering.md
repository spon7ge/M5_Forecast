# M5 Feature Engineering & Batch Model Training

This module implements a comprehensive feature engineering pipeline and LightGBM batch model training for M5 forecasting using Apache Spark DataFrames.

## Features

### 🔧 Feature Engineering
- **Time Window Aggregations**: 5-minute and 15-minute window aggregations
- **Lag Features**: Multiple lag periods (1, 2, 3, 6, 12, 24 intervals)
- **Rolling Statistics**: Rolling averages, standard deviation, min/max with various windows
- **Calendar Features**: Hour, day of week, month, quarter, weekends, month-end indicators
- **Target Creation**: Next time window demand prediction target

### 🤖 Machine Learning
- **LightGBM Regression**: Optimized gradient boosting for demand forecasting
- **Model Evaluation**: MAE, RMSE, MAPE metrics
- **Model Persistence**: Save/load models using joblib
- **Batch Prediction**: Production-ready prediction pipeline

### ⚡ Performance
- **Apache Spark**: Distributed processing for large-scale feature engineering
- **Optimized Configurations**: Adaptive query execution, skew join handling
- **Memory Efficient**: Streaming-based processing where possible

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements_ml.txt
```

### 2. Run Feature Engineering Pipeline

```bash
python run_feature_engineering.py
```

### 3. Use Trained Model for Predictions

```python
from batchPredictor import M5BatchPredictor
import pandas as pd

# Load trained model
predictor = M5BatchPredictor("models/m5_lightgbm_model_20250105_123456.pkl")

# Prepare your data
data = pd.DataFrame({
    'timestamp': ['2024-01-01 10:00:00', '2024-01-01 10:15:00'],
    'item_id': ['ITEM_0001', 'ITEM_0001'], 
    'store_id': ['CA_1', 'CA_1'],
    'demand': [15, 18]
})

# Make predictions
results = predictor.predict_batch(data)
print(results)
```

## Architecture Overview

### Data Flow

```
Raw Sales Data → Time Window Aggregation → Feature Engineering → Model Training → Saved Model
                                      ↓
                              Calendar Features
                              Lag Features  
                              Rolling Features
```

### Components

1. **M5FeatureEngineer**: Main pipeline class
2. **Time Aggregation**: Group data into 5min/15min windows
3. **Feature Creation**: Generate lag, rolling, and calendar features
4. **Model Training**: LightGBM with early stopping and validation
5. **Model Persistence**: Save models and metrics for production use

## Detailed Usage

### Feature Engineering Pipeline

```python
from featureEngineering import M5FeatureEngineer

# Initialize pipeline
engineer = M5FeatureEngineer()

# Create sample data (or load your own)
sales_df = engineer.create_sample_sales_data(num_items=1000, num_days=90)

# Run complete pipeline
model, metrics, model_files = engineer.run_complete_pipeline()
```

### Custom Feature Engineering

```python
# Step-by-step feature engineering
agg_5min, agg_15min = engineer.create_time_window_aggregations(sales_df)
df_with_lags = engineer.create_lag_features(agg_15min)
df_with_rolling = engineer.create_rolling_features(df_with_lags)
df_with_calendar = engineer.create_calendar_features(df_with_rolling)
df_with_target = engineer.create_target_variable(df_with_calendar)
```

### Model Training

```python
# Prepare data for ML
ml_df, feature_cols, target_col = engineer.prepare_features_for_ml(df_with_target)

# Train model
model, metrics, test_results = engineer.train_lightgbm_model(
    ml_df, feature_cols, target_col, test_size=0.2
)

# Save model
model_files = engineer.save_model(model, metrics)
```

## Configuration

### Spark Configuration
- Adaptive Query Execution enabled
- Skew join optimization
- Local shuffle reader optimization
- Kryo serialization

### LightGBM Parameters
- Objective: regression
- Metric: RMSE
- Learning rate: 0.1
- Early stopping: 100 rounds
- Feature/bagging fraction: 0.8

## Generated Features

### Time-based Features
- `hour_of_day`: 0-23
- `day_of_week`: 0-6 (Monday=0)
- `day_of_month`: 1-31
- `day_of_year`: 1-366
- `week_of_year`: 1-53
- `quarter`: 1-4
- `is_weekend`: 0/1
- `is_month_start/end`: 0/1
- `is_quarter_start/end`: 0/1

### Lag Features
- `demand_15min_total_lag_1`: Previous period demand
- `demand_15min_total_lag_2`: 2 periods ago
- `demand_15min_total_lag_3`: 3 periods ago
- `demand_15min_total_lag_6`: 6 periods ago (1.5 hours)
- `demand_15min_total_lag_12`: 12 periods ago (3 hours)
- `demand_15min_total_lag_24`: 24 periods ago (6 hours)

### Rolling Features
For windows [3, 6, 12, 24, 48]:
- `demand_15min_total_rolling_avg_{window}`: Rolling average
- `demand_15min_total_rolling_std_{window}`: Rolling standard deviation
- `demand_15min_total_rolling_min_{window}`: Rolling minimum
- `demand_15min_total_rolling_max_{window}`: Rolling maximum

### Aggregation Features
- `demand_15min_total`: Sum of demand in 15-min window
- `demand_15min_avg`: Average demand in window
- `demand_15min_count`: Number of observations
- `demand_15min_std`: Standard deviation
- `demand_15min_min/max`: Min/max values

## Performance Metrics

The pipeline evaluates models using:
- **MAE (Mean Absolute Error)**: Average absolute prediction error
- **RMSE (Root Mean Squared Error)**: Square root of mean squared error
- **MAPE (Mean Absolute Percentage Error)**: Percentage-based error metric

## File Structure

```
├── featureEngineering.py          # Main pipeline implementation
├── batchPredictor.py              # Batch prediction pipeline
├── run_feature_engineering.py     # Standalone execution script
├── FeatureEngineering_Demo.ipynb  # Interactive demonstration
├── requirements_ml.txt            # Python dependencies
├── models/                        # Saved models directory
│   ├── m5_lightgbm_model_*.pkl   # Trained models
│   └── m5_lightgbm_model_*_metrics.pkl # Model metrics
└── README_FeatureEngineering.md   # This documentation
```

## Sample Output

```
=== Model Performance ===
train_mae     : 2.3456
train_rmse    : 3.1234
test_mae      : 2.4567
test_rmse     : 3.2345

Feature columns (47): ['demand_15min_avg', 'demand_15min_count', 'demand_15min_std', ...]
ML dataset size: 125,847 records
Model training completed!
```

## Advanced Usage

### Custom Time Windows

```python
# Create custom aggregation windows
df_custom = sales_df.groupBy("item_id", "store_id", 
    window(col("timestamp"), "30 minutes")) \
    .agg(sum("demand").alias("demand_30min"))
```

### Additional Features

```python
# Add custom business logic features
df_enhanced = df.withColumn("is_business_hours", 
    when((col("hour") >= 9) & (col("hour") <= 17), 1).otherwise(0))
```

### Model Hyperparameter Tuning

```python
# Custom LightGBM parameters
custom_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'learning_rate': 0.05,
    'num_leaves': 63,
    'feature_fraction': 0.9
}
```

## Troubleshooting

### Common Issues

1. **Spark Memory Issues**: Increase driver/executor memory in spark config
2. **Missing Features**: Ensure sufficient historical data for lag features
3. **Null Values**: Check data quality, especially for time-based features

### Performance Optimization

1. **Partition Data**: Partition by item_id/store_id for better performance
2. **Cache DataFrames**: Cache intermediate results for iterative operations
3. **Broadcast Joins**: Use broadcast for small dimension tables

## Integration with Existing Pipeline

This feature engineering module integrates with the existing M5 streaming pipeline:

1. **Data Source**: Uses same input format as `sparkStreamingPipeline.py`
2. **Configuration**: Shares configuration from `config.py`
3. **Output Format**: Compatible with existing Delta Lake storage

## Future Enhancements

- [ ] Add more sophisticated calendar features (holidays, events)
- [ ] Implement seasonal decomposition features
- [ ] Add cross-validation for model selection
- [ ] Support for multiple prediction horizons
- [ ] Integration with MLflow for experiment tracking
- [ ] Real-time feature serving capabilities