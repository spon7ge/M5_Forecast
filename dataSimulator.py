# Data simulator to test the streaming pipeline

import pandas as pd
import numpy as np
import os
import time
from datetime import datetime, timedelta
import shutil

class M5DataSimulator:
    """Simulate M5 data files for testing the streaming pipeline"""
    
    def __init__(self, base_path="./input/"):
        self.base_path = base_path
        self.setup_directories()
        
    def setup_directories(self):
        """Create necessary directories"""
        dirs = ["calendar", "sales", "prices"]
        for dir_name in dirs:
            os.makedirs(os.path.join(self.base_path, dir_name), exist_ok=True)
    
    def simulate_calendar_files(self, num_files=5, records_per_file=100):
        """Simulate calendar data files arriving over time"""
        
        base_date = datetime(2011, 1, 29)
        
        for i in range(num_files):
            print(f"Generating calendar file {i+1}/{num_files}")
            
            # Generate date range for this file
            start_date = base_date + timedelta(days=i * records_per_file)
            dates = [start_date + timedelta(days=j) for j in range(records_per_file)]
            
            # Create sample data
            data = []
            for j, date in enumerate(dates):
                data.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'wm_yr_wk': int(f"{date.year}{date.isocalendar()[1]:02d}"),
                    'weekday': date.strftime('%A'),
                    'wday': date.isoweekday() % 7 + 1,  # Convert to M5 format
                    'month': date.month,
                    'year': date.year,
                    'd': f'd_{i * records_per_file + j + 1}',
                    'event_name_1': np.random.choice([None, 'Christmas', 'Thanksgiving', 'NewYear'], p=[0.9, 0.03, 0.03, 0.04]),
                    'event_type_1': np.random.choice([None, 'Religious', 'Cultural', 'National'], p=[0.9, 0.03, 0.03, 0.04]),
                    'event_name_2': None,
                    'event_type_2': None,
                    'snap_CA': np.random.choice([0, 1], p=[0.7, 0.3]),
                    'snap_TX': np.random.choice([0, 1], p=[0.7, 0.3]),
                    'snap_WI': np.random.choice([0, 1], p=[0.7, 0.3])
                })
            
            # Create DataFrame and save
            df = pd.DataFrame(data)
            filename = f"calendar_batch_{i+1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            filepath = os.path.join(self.base_path, "calendar", filename)
            df.to_csv(filepath, index=False)
            
            print(f"Created: {filepath}")
            time.sleep(2)  # Simulate files arriving over time
    
    def simulate_sales_files(self, num_files=3, records_per_file=50):
        """Simulate sales data files"""
        
        stores = ['CA_1', 'CA_2', 'TX_1', 'TX_2', 'WI_1']
        categories = ['HOBBIES', 'HOUSEHOLD', 'FOODS']
        departments = ['HOBBIES_1', 'HOUSEHOLD_1', 'FOODS_1', 'FOODS_2']
        
        for i in range(num_files):
            print(f"Generating sales file {i+1}/{num_files}")
            
            data = []
            for j in range(records_per_file):
                # Create base item info
                item_data = {
                    'item_id': f'ITEM_{i}_{j:03d}',
                    'dept_id': np.random.choice(departments),
                    'cat_id': np.random.choice(categories),
                    'store_id': np.random.choice(stores),
                    'state_id': np.random.choice(['CA', 'TX', 'WI'])
                }
                
                # Add daily sales (d_1, d_2, etc.)
                for day in range(1, 101):  # 100 days of sales
                    item_data[f'd_{day}'] = max(0, int(np.random.poisson(5)))
                
                data.append(item_data)
            
            df = pd.DataFrame(data)
            filename = f"sales_batch_{i+1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            filepath = os.path.join(self.base_path, "sales", filename)
            df.to_csv(filepath, index=False)
            
            print(f"Created: {filepath}")
            time.sleep(3)
    
    def simulate_prices_files(self, num_files=2, records_per_file=30):
        """Simulate prices data files"""
        
        stores = ['CA_1', 'CA_2', 'TX_1', 'TX_2', 'WI_1']
        categories = ['HOBBIES', 'HOUSEHOLD', 'FOODS']
        departments = ['HOBBIES_1', 'HOUSEHOLD_1', 'FOODS_1', 'FOODS_2']
        
        for i in range(num_files):
            print(f"Generating prices file {i+1}/{num_files}")
            
            data = []
            for j in range(records_per_file):
                # Generate realistic price data
                base_price = np.random.uniform(5.0, 50.0)  # Base price between $5-$50
                data.append({
                    'store_id': np.random.choice(stores),
                    'item_id': f'ITEM_{i}_{j:03d}',
                    'wm_yr_wk': np.random.randint(11101, 11600),  # Week numbers
                    'sell_price': round(base_price + np.random.normal(0, 2), 2)  # Price with some variation
                })
            
            df = pd.DataFrame(data)
            filename = f"prices_batch_{i+1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            filepath = os.path.join(self.base_path, "prices", filename)
            df.to_csv(filepath, index=False)
            
            print(f"Created: {filepath}")
            time.sleep(3)
    
    def clean_input_directories(self):
        """Clean up input directories"""
        dirs = ["calendar", "sales", "prices"]
        for dir_name in dirs:
            dir_path = os.path.join(self.base_path, dir_name)
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path)
            os.makedirs(dir_path, exist_ok=True)
        print("Input directories cleaned.")

# Test script
if __name__ == "__main__":
    simulator = M5DataSimulator()
    
    # Clean previous test data
    simulator.clean_input_directories()
    
    # Simulate calendar files
    simulator.simulate_calendar_files(num_files=3, records_per_file=50)