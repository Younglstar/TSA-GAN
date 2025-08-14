# normalizer_base.py
# Abstract base class defining the interface for normalizers
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Dict, Union, Tuple
from normalizer_config import NormalizerConfig

class BaseNormalizer(ABC):
    def __init__(self, config: NormalizerConfig):
        self.config = config
        self.params = {}
        np.random.seed(config.RANDOM_SEED)
        
    @abstractmethod
    def fit(self, data: Union[pd.DataFrame, pd.Series]) -> None:
        """Fit normalizer to the data"""
        pass
    
    @abstractmethod
    def transform(self, data: Union[pd.DataFrame, pd.Series]) -> Union[pd.DataFrame, pd.Series]:
        """Transform data using fitted parameters"""
        pass
    
    @abstractmethod
    def inverse_transform(self, data: Union[pd.DataFrame, pd.Series]) -> Union[pd.DataFrame, pd.Series]:
        """Inverse transform normalized data back to original scale"""
        pass
    
    def fit_transform(self, data: Union[pd.DataFrame, pd.Series]) -> Union[pd.DataFrame, pd.Series]:
        """Fit and transform data"""
        self.fit(data)
        return self.transform(data)
    
    def save_params(self, feature_name: str) -> None:
        """Save normalization parameters"""
        if self.config.normalization_params is None:
            self.config.normalization_params = {}
        self.config.normalization_params[feature_name] = self.params
    
    def load_params(self, feature_name: str) -> None:
        """Load normalization parameters"""
        if self.config.normalization_params is not None:
            self.params = self.config.normalization_params.get(feature_name, {})