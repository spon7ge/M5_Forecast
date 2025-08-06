from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
import os
from datetime import datetime

class M5DataIngestionPipeline:
    def __init__(self, app_name="M5-Data-Ingestion"):
        """Initialize Spark session with optimized configurations for streaming"""
        self.spark = SparkSession.builder \
            .appName(app_name) \
            .config("spark.sql.streaming.checkpointLocation", "./checkpoints") \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
            .config("spark.driver.bindAddress", "127.0.0.1") \
            .config("spark.driver.host", "127.0.0.1") \
            .config("spark.ui.enabled", "false") \
            .getOrCreate()
        
        # Set log level to reduce noise
        self.spark.sparkContext.setLogLevel("WARN")
        
    def define_schemas(self):
        """Define schemas for different M5 data sources"""
        
        # Calendar data schema
        self.calendar_schema = StructType([
            StructField("date", StringType(), True),
            StructField("wm_yr_wk", IntegerType(), True),
            StructField("weekday", StringType(), True),
            StructField("wday", IntegerType(), True),
            StructField("month", IntegerType(), True),
            StructField("year", IntegerType(), True),
            StructField("d", StringType(), True),
            StructField("event_name_1", StringType(), True),
            StructField("event_type_1", StringType(), True),
            StructField("event_name_2", StringType(), True),
            StructField("event_type_2", StringType(), True),
            StructField("snap_CA", IntegerType(), True),
            StructField("snap_TX", IntegerType(), True),
            StructField("snap_WI", IntegerType(), True)
        ])
        
        # Sales data schema (for when you have sales files)
        self.sales_schema = StructType([
            StructField("item_id", StringType(), True),
            StructField("dept_id", StringType(), True),
            StructField("cat_id", StringType(), True),
            StructField("store_id", StringType(), True),
            StructField("state_id", StringType(), True)
        ])
        
        # Prices data schema
        self.prices_schema = StructType([
            StructField("store_id", StringType(), True),
            StructField("item_id", StringType(), True),
            StructField("wm_yr_wk", IntegerType(), True),
            StructField("sell_price", DoubleType(), True)
        ])
        
    def create_file_stream(self, input_path, schema, data_type="csv"):
        """Create a file-based streaming DataFrame"""
        
        if data_type == "csv":
            return self.spark \
                .readStream \
                .option("header", "true") \
                .option("inferSchema", "false") \
                .schema(schema) \
                .csv(input_path)
        elif data_type == "json":
            return self.spark \
                .readStream \
                .schema(schema) \
                .json(input_path)
        elif data_type == "parquet":
            return self.spark \
                .readStream \
                .schema(schema) \
                .parquet(input_path)
        else:
            raise ValueError(f"Unsupported data type: {data_type}")
    
    def transform_calendar_data(self, df):
        """Apply transformations to calendar data stream"""
        
        return df.select(
            to_date(col("date"), "yyyy-MM-dd").alias("date"),
            col("wm_yr_wk"),
            col("weekday"),
            col("wday"),
            col("month"),
            col("year"),
            col("d"),
            col("event_name_1"),
            col("event_type_1"),
            col("event_name_2"),
            col("event_type_2"),
            col("snap_CA").cast("boolean").alias("snap_CA"),
            col("snap_TX").cast("boolean").alias("snap_TX"),
            col("snap_WI").cast("boolean").alias("snap_WI"),
            # Add derived features
            dayofweek(to_date(col("date"), "yyyy-MM-dd")).alias("day_of_week"),
            dayofyear(to_date(col("date"), "yyyy-MM-dd")).alias("day_of_year"),
            quarter(to_date(col("date"), "yyyy-MM-dd")).alias("quarter"),
            when(col("wday").isin([1, 2]), True).otherwise(False).alias("is_weekend"),
            when(col("event_name_1").isNotNull() | col("event_name_2").isNotNull(), True)
                .otherwise(False).alias("has_event"),
            # Processing metadata
            current_timestamp().alias("processed_timestamp"),
            lit("calendar").alias("data_source")
        )
    
    def transform_sales_data(self, df):
        """Apply transformations to sales data stream"""
        
        # Extract day columns dynamically (d_1, d_2, etc.)
        day_columns = [col_name for col_name in df.columns if col_name.startswith('d_')]
        
        # Unpivot the sales data from wide to long format
        sales_long = df.select(
            col("item_id"),
            col("dept_id"), 
            col("cat_id"),
            col("store_id"),
            col("state_id"),
            # Stack day columns
            stack(
                len(day_columns),
                *[lit(day_col) for day_col in day_columns] + 
                 [col(day_col) for day_col in day_columns]
            ).alias("day", "sales_quantity")
        )
        
        return sales_long.select(
            col("item_id"),
            col("dept_id"),
            col("cat_id"), 
            col("store_id"),
            col("state_id"),
            col("day"),
            col("sales_quantity").cast("integer"),
            # Add processing metadata
            current_timestamp().alias("processed_timestamp"),
            lit("sales").alias("data_source")
        )
    
    def transform_prices_data(self, df):
        """Apply transformations to prices data stream"""
        
        return df.select(
            col("store_id"),
            col("item_id"),
            col("wm_yr_wk"),
            col("sell_price"),
            # Add derived features
            when(col("sell_price").isNull(), True).otherwise(False).alias("price_missing"),
            # Processing metadata
            current_timestamp().alias("processed_timestamp"),
            lit("prices").alias("data_source")
        )
    
    def write_to_delta(self, df, output_path, checkpoint_path, table_name=None):
        """Write streaming DataFrame to Delta Lake format"""
        
        writer = df.writeStream \
            .format("delta") \
            .outputMode("append") \
            .option("checkpointLocation", checkpoint_path) \
            .option("path", output_path)
        
        if table_name:
            writer = writer.option("tableName", table_name)
            
        return writer
    
    def write_to_console(self, df, truncate=False):
        """Write streaming DataFrame to console for debugging"""
        
        return df.writeStream \
            .outputMode("append") \
            .format("console") \
            .option("truncate", truncate) \
            .option("numRows", 20)
    
    def write_to_memory(self, df, query_name):
        """Write streaming DataFrame to memory table for interactive queries"""
        
        return df.writeStream \
            .queryName(query_name) \
            .outputMode("append") \
            .format("memory")
    
    def create_calendar_pipeline(self, input_path="./input/calendar/", 
                                output_path="./output/calendar_delta/"):
        """Create calendar data ingestion pipeline"""
        
        print("Starting calendar data pipeline...")
        
        # Create input stream
        calendar_stream = self.create_file_stream(
            input_path, 
            self.calendar_schema, 
            "csv"
        )
        
        # Apply transformations
        calendar_transformed = self.transform_calendar_data(calendar_stream)
        
        # Write to Delta Lake
        calendar_query = self.write_to_delta(
            calendar_transformed,
            output_path,
            "./checkpoints/calendar",
            "calendar_data"
        ).start()
        
        return calendar_query
    
    def create_sales_pipeline(self, input_path="./input/sales/",
                             output_path="./output/sales_delta/"):
        """Create sales data ingestion pipeline"""
        
        print("Starting sales data pipeline...")
        
        # Note: For sales data, you'd need to modify the schema based on actual columns
        sales_stream = self.create_file_stream(
            input_path,
            self.sales_schema,  # You'd need to extend this with d_1, d_2, etc. columns
            "csv"
        )
        
        sales_transformed = self.transform_sales_data(sales_stream)
        
        sales_query = self.write_to_delta(
            sales_transformed,
            output_path,
            "./checkpoints/sales",
            "sales_data"
        ).start()
        
        return sales_query
    
    def create_prices_pipeline(self, input_path="./input/prices/",
                              output_path="./output/prices_delta/"):
        """Create prices data ingestion pipeline"""
        
        print("Starting prices data pipeline...")
        
        prices_stream = self.create_file_stream(
            input_path,
            self.prices_schema,
            "csv"
        )
        
        prices_transformed = self.transform_prices_data(prices_stream)
        
        prices_query = self.write_to_delta(
            prices_transformed,
            output_path,
            "./checkpoints/prices",
            "prices_data"
        ).start()
        
        return prices_query
    
    def monitor_streams(self, queries):
        """Monitor multiple streaming queries"""
        
        print("Monitoring streaming queries...")
        print(f"Active queries: {len(queries)}")
        
        for i, query in enumerate(queries):
            print(f"Query {i+1}: {query.name} - Status: {query.status}")
            
    def stop_all_streams(self, queries):
        """Stop all streaming queries gracefully"""
        
        print("Stopping all streaming queries...")
        for query in queries:
            query.stop()
        print("All queries stopped.")
    
    def create_monitoring_dashboard(self):
        """Create a simple monitoring dashboard"""
        
        dashboard_query = """
        SELECT 
            data_source,
            COUNT(*) as record_count,
            MAX(processed_timestamp) as last_processed,
            MIN(processed_timestamp) as first_processed
        FROM global_temp.monitoring_view
        GROUP BY data_source
        ORDER BY data_source
        """
        
        return dashboard_query

# Usage example and configuration
def main():
    """Main execution function"""
    
    # Initialize the pipeline
    pipeline = M5DataIngestionPipeline("M5-Forecasting-Pipeline")
    pipeline.define_schemas()
    
    # Create directories if they don't exist
    os.makedirs("./input/calendar", exist_ok=True)
    os.makedirs("./output/calendar_delta", exist_ok=True)
    os.makedirs("./checkpoints", exist_ok=True)
    
    active_queries = []
    
    try:
        # Start calendar pipeline
        calendar_query = pipeline.create_calendar_pipeline()
        active_queries.append(calendar_query)
        
        # You can add other pipelines as needed
        # sales_query = pipeline.create_sales_pipeline()
        # prices_query = pipeline.create_prices_pipeline()
        # active_queries.extend([sales_query, prices_query])
        
        # Monitor the streams
        pipeline.monitor_streams(active_queries)
        
        # Keep the streams running
        print("Streams are running. Press Ctrl+C to stop...")
        for query in active_queries:
            query.awaitTermination()
            
    except KeyboardInterrupt:
        print("\nShutdown requested...")
        pipeline.stop_all_streams(active_queries)
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        pipeline.stop_all_streams(active_queries)
    finally:
        pipeline.spark.stop()

if __name__ == "__main__":
    main()