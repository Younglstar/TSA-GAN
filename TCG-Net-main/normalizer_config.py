# normalizer_config.py
# Configuration class defining numerical features and parameters
from dataclasses import dataclass, field
from typing import List, Dict

def get_static_numerical_features() -> List[str]:
    return [
        'AGE_ARRIVEE',
        'GP_ONFILE'
    ]

def get_temporal_numerical_features() -> List[str]:
    return [
        'DUREE_SEJOUR_HOUR'
    ]

@dataclass
class NormalizerConfig:
    # Numerical features to normalize
    STATIC_NUMERICAL_FEATURES: List[str] = field(
        default_factory=get_static_numerical_features
    )
    
    TEMPORAL_NUMERICAL_FEATURES: List[str] = field(
        default_factory=get_temporal_numerical_features
    )
    
    # Parameters for normalization
    RANDOM_SEED: int = 42
    EPSILON: float = 1e-10  # Small value to avoid division by zero
    
    # File paths
    NORMALIZATION_PARAMS_FILE: str = 'normalization_params.pkl'
    
    # Dictionary to store normalization parameters
    normalization_params: Dict = field(default_factory=dict)