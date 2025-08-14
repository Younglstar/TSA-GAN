# embedding_config.py
from dataclasses import dataclass, field
from typing import List, Dict

def get_static_categorical_features() -> List[str]:
    return [
        'SEXE',
        'MARITAL_STATUS',
        'GP_ONFILE'  # If this is categorical
    ]

def get_temporal_categorical_features() -> List[str]:
    return [
        'ETABLISSEMENT',
        'UNITE',
        'Diagnostic',
        'RAISON_VISITE'
    ]

@dataclass
class EmbeddingConfig:
    # Categorical features for embedding
    STATIC_CATEGORICAL_FEATURES: List[str] = field(
        default_factory=get_static_categorical_features
    )
    
    TEMPORAL_CATEGORICAL_FEATURES: List[str] = field(
        default_factory=get_temporal_categorical_features
    )
    
    # Embedding dimensions (can be adjusted based on cardinality)
    DEFAULT_EMBEDDING_DIM: int = 8
    MIN_EMBEDDING_DIM: int = 4
    MAX_EMBEDDING_DIM: int = 32
    
    # Training parameters
    BATCH_SIZE: int = 64
    LEARNING_RATE: float = 0.001
    NUM_EPOCHS: int = 100
    
    # Model parameters
    HIDDEN_LAYERS: List[int] = field(default_factory=lambda: [64, 32])
    DROPOUT_RATE: float = 0.2
    
    # File paths
    EMBEDDING_MODELS_FILE: str = 'embedding_models.pkl'
    EMBEDDINGS_FILE: str = 'categorical_embeddings.pkl'
    
    # Dictionary to store category mappings
    category_mappings: Dict = field(default_factory=dict)