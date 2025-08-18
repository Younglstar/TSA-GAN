# stochastic_normalizer.py
# Implementation of stochastic normalization algorithm
import numpy as np
import pandas as pd
from typing import Dict, Union, Tuple
from normalizer_base import BaseNormalizer
from normalizer_config import NormalizerConfig

class StochasticNormalizer(BaseNormalizer):
    def __init__(self, config: NormalizerConfig):
        super().__init__(config)
        
    def _calculate_distribution_params(self, data: Union[pd.Series, np.ndarray]) -> Dict:
        """Calculate distribution parameters for stochastic normalization"""
        # Convert to numpy array if needed
        values = data.values if isinstance(data, pd.Series) else data
        
        # Get unique values and their frequencies
        unique_vals, counts = np.unique(values, return_counts=True)
        frequencies = counts / len(values)
        
        # Calculate cumulative probabilities
        cum_probs = np.cumsum(frequencies)
        
        return {
            'unique_values': unique_vals,
            'frequencies': frequencies,
            'cum_probs': cum_probs,
            'bounds': np.concatenate(([0], cum_probs))
        }
    
    def fit(self, data: Union[pd.DataFrame, pd.Series]) -> None:
        """Fit the stochastic normalizer to the data"""
        self.params = self._calculate_distribution_params(data)
    
    def transform(self, data: Union[pd.DataFrame, pd.Series]) -> Union[pd.DataFrame, pd.Series]:
        """Transform data using stochastic normalization"""
        values = data.values if isinstance(data, pd.Series) else data
        transformed = np.zeros_like(values, dtype=float)
        
        for i, val in enumerate(values):
            # Find the corresponding bin for the value
            idx = np.where(self.params['unique_values'] == val)[0][0]
            
            # Generate random value within the bin's bounds
            lower_bound = self.params['bounds'][idx]
            upper_bound = self.params['bounds'][idx + 1]
            
            transformed[i] = np.random.uniform(lower_bound, upper_bound)
        
        if isinstance(data, pd.Series):
            return pd.Series(transformed, index=data.index)
        return transformed
    
    def inverse_transform(self, data: Union[pd.DataFrame, pd.Series]) -> Union[pd.DataFrame, pd.Series]:
        """Inverse transform normalized data back to original scale"""
        values = data.values if isinstance(data, pd.Series) else data
        inverse_transformed = np.zeros_like(values)
        
        for i, val in enumerate(values):
            # Find the bin that contains the normalized value
            bin_idx = np.digitize(val, self.params['bounds']) - 1
            
            # Map back to original value
            inverse_transformed[i] = self.params['unique_values'][bin_idx]
        
        if isinstance(data, pd.Series):
            return pd.Series(inverse_transformed, index=data.index)
        return inverse_transformed