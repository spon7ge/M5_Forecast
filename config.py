# Configuration file for M5 Data Ingestion Pipeline

import os
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class StreamingConfig:
    """Configuration class for streaming parameters"""
    
    # Spark configurations
    spark_configs: Dict[str, str] = None
    
    # Input paths
    calendar_input_path: str = "./input/calendar/"
    sales_input_path: str = "./input/sales/"
    prices_input_path: str = "./input/prices/"
    
    # Output paths
    calendar_output_path: str = "./output/calendar_delta/"
    sales_output_path: str = "./output/sales_delta/"
    prices_output_path: str = "./output/prices_delta/"
    
    # Checkpoint paths
    checkpoint_base_path: str = "./checkpoints/"
    
    # Processing configurations
    trigger_interval: str = "10 seconds"
    max_files_per_trigger: int = 10
    
    # Monitoring
    enable_monitoring: bool = True
    monitoring_interval: int = 30  # seconds
    
    def __post_init__(self):
        if self.spark_configs is None:
            self.spark_configs = {
                "spark.sql.streaming.checkpointLocation": self.checkpoint_base_path,
                "spark.sql.adaptive.enabled": "true",
                "spark.sql.adaptive.coalescePartitions.enabled": "true",
                "spark.serializer": "org.apache.spark.serializer.KryoSerializer",
                "spark.sql.streaming.maxFilesPerTrigger": str(self.max_files_per_trigger),
                "spark.sql.streaming.fileSource.log.deletion": "true",
                "spark.sql.streaming.fileSource.log.compactInterval": "10",
                "spark.sql.streaming.fileSource.log.cleanupDelay": "2000"
            }

# Environment-specific configurations
CONFIGS = {
    "development": StreamingConfig(
        trigger_interval="5 seconds",
        max_files_per_trigger=5,
        enable_monitoring=True
    ),
    "production": StreamingConfig(
        trigger_interval="30 seconds", 
        max_files_per_trigger=50,
        enable_monitoring=True,
        monitoring_interval=60
    ),
    "testing": StreamingConfig(
        trigger_interval="1 second",
        max_files_per_trigger=1,
        enable_monitoring=False
    )
}

def get_config(environment: str = "development") -> StreamingConfig:
    """Get configuration for specified environment"""
    return CONFIGS.get(environment, CONFIGS["development"])