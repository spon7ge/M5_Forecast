# Main runner script for the M5 data ingestion pipeline

import sys
import argparse
from sparkStreamingPipeline import M5DataIngestionPipeline
from config import get_config
from dataSimulator import M5DataSimulator
import threading
import time

def run_pipeline(environment="development", simulate_data=False):
    """Run the M5 data ingestion pipeline"""
    
    # Get configuration
    config = get_config(environment)
    
    # Initialize pipeline
    pipeline = M5DataIngestionPipeline("M5-Forecasting-Pipeline")
    pipeline.define_schemas()
    
    active_queries = []
    
    try:
        # Start data simulation if requested
        if simulate_data:
            print("Starting data simulation...")
            simulator = M5DataSimulator()
            
            # Start calendar simulation
            calendar_thread = threading.Thread(
                target=simulator.simulate_calendar_files,
                kwargs={"num_files": 5, "records_per_file": 100}
            )
            calendar_thread.daemon = True
            calendar_thread.start()
            
            # Start sales simulation
            sales_thread = threading.Thread(
                target=simulator.simulate_sales_files,
                kwargs={"num_files": 3, "records_per_file": 50}
            )
            sales_thread.daemon = True
            sales_thread.start()
            
            # Start prices simulation
            prices_thread = threading.Thread(
                target=simulator.simulate_prices_files,
                kwargs={"num_files": 2, "records_per_file": 30}
            )
            prices_thread.daemon = True
            prices_thread.start()
            
            time.sleep(2)  # Give simulator time to create first files
        
        # Start calendar pipeline
        print("Starting calendar data pipeline...")
        calendar_query = pipeline.create_calendar_pipeline(
            input_path=config.calendar_input_path,
            output_path=config.calendar_output_path
        )
        active_queries.append(calendar_query)
        
        # Add monitoring if enabled
        if config.enable_monitoring:
            print("Monitoring enabled...")
            def monitor_loop():
                while True:
                    pipeline.monitor_streams(active_queries)
                    time.sleep(config.monitoring_interval)
            
            monitor_thread = threading.Thread(target=monitor_loop)
            monitor_thread.daemon = True
            monitor_thread.start()
        
        print(f"Pipeline running in {environment} mode...")
        print("Press Ctrl+C to stop...")
        
        # Keep streams running
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

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="M5 Data Ingestion Pipeline")
    parser.add_argument("--env", default="development", 
                       choices=["development", "production", "testing"],
                       help="Environment to run in")
    parser.add_argument("--simulate", action="store_true",
                       help="Generate simulated data for testing")
    
    args = parser.parse_args()
    
    run_pipeline(args.env, args.simulate)

if __name__ == "__main__":
    main()