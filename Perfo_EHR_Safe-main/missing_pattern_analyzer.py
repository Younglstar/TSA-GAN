# missing_pattern_analyzer.py
import pandas as pd
import numpy as np
from typing import Dict, Tuple

class MissingPatternAnalyzer:
    def __init__(self, static_data: pd.DataFrame, temporal_data: Dict[str, pd.DataFrame]):
        self.static_data = static_data
        self.temporal_data = temporal_data
        self.static_missing_rates = None
        self.temporal_missing_rates = None
        self.missing_patterns = None
        
    def calculate_missing_rates(self) -> Tuple[pd.Series, Dict[str, pd.Series]]:
        """Calculate missing rates for static and temporal features"""
        # Static missing rates
        self.static_missing_rates = (self.static_data.isna().sum() / len(self.static_data))
        
        # Temporal missing rates for each visit type
        self.temporal_missing_rates = {
            visit_type: (data.isna().sum() / len(data))
            for visit_type, data in self.temporal_data.items()
        }
        
        return self.static_missing_rates, self.temporal_missing_rates
    
    def generate_missing_patterns(self) -> Dict[str, pd.DataFrame]:
        """Generate binary missing patterns for all features"""
        # Static patterns
        static_patterns = self.static_data.isna().astype(int)
        
        # Temporal patterns for each visit
        temporal_patterns = {
            visit_type: data.isna().astype(int)
            for visit_type, data in self.temporal_data.items()
        }
        
        self.missing_patterns = {
            'static': static_patterns,
            'temporal': temporal_patterns
        }
        
        return self.missing_patterns