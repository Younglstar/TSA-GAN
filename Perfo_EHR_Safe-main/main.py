# main.py
import pandas as pd
import numpy as np
from config import DataConfig
from data_loader import DataLoader
from missing_pattern_analyzer import MissingPatternAnalyzer
import pickle
import os

def main():
    # Initialize configuration
    config = DataConfig()
    
    # Initialize data loader
    loader = DataLoader(config)
    
    print("Loading and preprocessing data...")
    # Load and preprocess data
    data = loader.load_data()
    static_data, temporal_data = loader.split_static_temporal()
    
    print("Analyzing missing patterns...")
    # Analyze missing patterns
    analyzer = MissingPatternAnalyzer(static_data, temporal_data)
    missing_rates_static, missing_rates_temporal = analyzer.calculate_missing_rates()
    missing_patterns = analyzer.generate_missing_patterns()
    
    # Print some basic statistics
    print("\nBasic Statistics:")
    print(f"Number of patients: {len(static_data)}")
    print("\nMissing rates for static features:")
    print(missing_rates_static)
    print("\nMissing rates for temporal features (initial visit):")
    print(missing_rates_temporal['initial'])
    
    print("\nSaving processed data...")
    # Save processed data for next steps
    processed_data = {
        'static_data': static_data,
        'temporal_data': temporal_data,
        'missing_patterns': missing_patterns,
        'missing_rates': {
            'static': missing_rates_static,
            'temporal': missing_rates_temporal
        }
    }
    
    with open(config.PROCESSED_FILE, 'wb') as f:
        pickle.dump(processed_data, f)
    
    print(f"\nProcessed data saved to {config.PROCESSED_FILE}")
    return processed_data

if __name__ == "__main__":
    processed_data = main()