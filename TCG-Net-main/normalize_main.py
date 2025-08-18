# normalize_main.py
#  Main script to run the normalization process
import pickle
import os
from normalizer_config import NormalizerConfig
from normalizer_processor import DataNormalizationProcessor

def main():
    print("Starting data normalization process...")
    
    # Check if preprocessed data exists
    if not os.path.exists('processed_data.pkl'):
        raise FileNotFoundError("processed_data.pkl not found. Please run preprocessing first.")
    
    # Load preprocessed data
    print("Loading preprocessed data...")
    with open('processed_data.pkl', 'rb') as f:
        processed_data = pickle.load(f)
    
    # Initialize configuration and processor
    config = NormalizerConfig()
    processor = DataNormalizationProcessor(config)
    
    # Normalize static features
    print("\nNormalizing static features...")
    normalized_static = processor.normalize_static_features(processed_data['static_data'])
    print("Completed static feature normalization.")
    
    # Normalize temporal features
    print("\nNormalizing temporal features...")
    normalized_temporal = processor.normalize_temporal_features(processed_data['temporal_data'])
    print("Completed temporal feature normalization.")
    
    # Save normalization parameters
    print("\nSaving normalization parameters...")
    processor.save_normalization_params()
    
    # Save normalized data
    print("\nSaving normalized data...")
    normalized_data = {
        'static_data': normalized_static,
        'temporal_data': normalized_temporal,
        'missing_patterns': processed_data['missing_patterns'],
        'missing_rates': processed_data['missing_rates']
    }
    
    with open('normalized_data.pkl', 'wb') as f:
        pickle.dump(normalized_data, f)
    
    print("\nNormalization complete!")
    print(f"- Normalized data saved to: normalized_data.pkl")
    print(f"- Normalization parameters saved to: {config.NORMALIZATION_PARAMS_FILE}")
    
    return normalized_data

if __name__ == "__main__":
    normalized_data = main()