# data_loader.py
import pandas as pd
import numpy as np
from typing import Tuple, Dict
from config import DataConfig

class DataLoader:
    def __init__(self, config: DataConfig):
        self.config = config
        self.data = None
        self.static_data = None
        self.temporal_data = None
        
    def load_data(self) -> pd.DataFrame:
        """Load and perform initial preprocessing of the data"""
        self.data = pd.read_csv(self.config.INPUT_FILE)
        
        # Convert date columns to datetime
        for date_col in self.config.DATE_COLUMNS:
            self.data[date_col] = pd.to_datetime(self.data[date_col])
            
        return self.data
    
    def split_static_temporal(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into static and temporal features"""
        if self.data is None:
            self.load_data()
            
        # Extract static features
        self.static_data = self.data[self.config.STATIC_FEATURES].copy()
        
        # Extract temporal features
        temporal_initial = {}
        temporal_followup = {}
        
        # Process initial visit features
        for feature in self.config.TEMPORAL_FEATURES:
            temporal_initial[feature] = self.data[feature].copy()
            followup_col = f"{feature}_Fol_Urg"
            if followup_col in self.data.columns:
                temporal_followup[feature] = self.data[followup_col].copy()
        
        # Convert to DataFrames
        self.temporal_data = {
            'initial': pd.DataFrame(temporal_initial),
            'followup': pd.DataFrame(temporal_followup)
        }
        
        return self.static_data, self.temporal_data