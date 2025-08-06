#!/bin/bash

# M5 Data Ingestion Pipeline Runner Script

echo "Starting M5 Data Ingestion Pipeline..."

# Set environment variables
export PYSPARK_PYTHON=python3
export JAVA_HOME=${JAVA_HOME:-/usr/lib/jvm/java-11-openjdk}

# Create necessary directories
mkdir -p input/calendar input/sales input/prices
mkdir -p output/calendar_delta output/sales_delta output/prices_delta
mkdir -p checkpoints logspy 

# Install dependencies if needed
if [ ! -f "requirements_installed.flag" ]; then
    echo "Installing Python dependencies..."
    pip install -r requirements.txt
    touch requirements_installed.flag
fi

# Run the pipeline
python pipelineRunner.py --env development --simulate

echo "Pipeline execution completed."