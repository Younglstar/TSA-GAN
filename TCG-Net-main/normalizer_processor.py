# normalizer_processor.py
# Processor class to handle normalization of all features
import pandas as pd
import pickle
from typing import Dict, Tuple
from normalizer_config import NormalizerConfig
from stochastic_normalizer import StochasticNormalizer

class DataNormalizationProcessor:
    def __init__(self, config: NormalizerConfig):
        self.config = config
        self.normalizers = {}
        
    def normalize_static_features(self, static_data: pd.DataFrame) -> pd.DataFrame:
        """Normalize static numerical features"""
        normalized_static = static_data.copy()
        
        for feature in self.config.STATIC_NUMERICAL_FEATURES:
            if feature in static_data.columns:
                normalizer = StochasticNormalizer(self.config)
                normalized_static[feature] = normalizer.fit_transform(static_data[feature])
                self.normalizers[f'static_{feature}'] = normalizer
                normalizer.save_params(f'static_{feature}')
                
        return normalized_static
    
    def normalize_temporal_features(self, temporal_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Normalize temporal numerical features"""
        normalized_temporal = {}
        
        for visit_type, data in temporal_data.items():
            normalized_data = data.copy()
            
            for feature in self.config.TEMPORAL_NUMERICAL_FEATURES:
                if feature in data.columns:
                    normalizer = StochasticNormalizer(self.config)
                    normalized_data[feature] = normalizer.fit_transform(data[feature])
                    self.normalizers[f'temporal_{visit_type}_{feature}'] = normalizer
                    normalizer.save_params(f'temporal_{visit_type}_{feature}')
            
            normalized_temporal[visit_type] = normalized_data
            
        return normalized_temporal
    
    def save_normalization_params(self) -> None:
        """Save normalization parameters to file"""
        with open(self.config.NORMALIZATION_PARAMS_FILE, 'wb') as f:
            pickle.dump(self.config.normalization_params, f)
    
    def load_normalization_params(self) -> None:
        """Load normalization parameters from file"""
        with open(self.config.NORMALIZATION_PARAMS_FILE, 'rb') as f:
            self.config.normalization_params = pickle.load(f)