# embedding_main.py
import pickle
import os
from embedding_config import EmbeddingConfig
from embedding_processor import CategoricalEmbeddingProcessor

def main():
    print("Starting categorical embedding process...")
    
    # Check if normalized data exists
    if not os.path.exists('normalized_data.pkl'):
        raise FileNotFoundError("normalized_data.pkl not found. Please run normalization first.")
    
    # Load normalized data
    print("Loading normalized data...")
    with open('normalized_data.pkl', 'rb') as f:
        normalized_data = pickle.load(f)
    
    # Initialize configuration and processor
    config = EmbeddingConfig()
    processor = CategoricalEmbeddingProcessor(config)
    
    # Process static features
    print("\nProcessing static categorical features...")
    static_embeddings = processor.fit_transform_static_features(
        normalized_data['static_data']
    )
    
    # Process temporal features
    print("\nProcessing temporal categorical features...")
    temporal_embeddings = processor.fit_transform_temporal_features(
        normalized_data['temporal_data']
    )
    
    # Save embedding models and parameters
    print("\nSaving embedding models and parameters...")
    processor.save_models()
    
    # Save embeddings
    print("Saving generated embeddings...")
    embeddings_data = {
        'static_embeddings': static_embeddings,
        'temporal_embeddings': temporal_embeddings,
        'missing_patterns': normalized_data['missing_patterns'],
        'missing_rates': normalized_data['missing_rates']
    }
    
    with open(config.EMBEDDINGS_FILE, 'wb') as f:
        pickle.dump(embeddings_data, f)
    
    print("\nEmbedding process complete!")
    print(f"- Embedding models saved to: {config.EMBEDDING_MODELS_FILE}")
    print(f"- Generated embeddings saved to: {config.EMBEDDINGS_FILE}")
    
    return embeddings_data

if __name__ == "__main__":
    embeddings_data = main()