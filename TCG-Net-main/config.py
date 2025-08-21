# config.py
from dataclasses import dataclass, field
from typing import List

def get_static_features() -> List[str]:
    return [

    ]

def get_temporal_features() -> List[str]:
    return [
        'Name','Open', 'High', 'Low', 'Close', 'Volume'
    ]

def get_date_columns() -> List[str]:
    return ['Date']

def get_id_columns() -> List[str]:
    return ['Name']
@dataclass
class DataConfig:
    # Static features configuration
    STATIC_FEATURES: List[str] = field(default_factory=get_static_features)
    
    # Temporal features configuration
    TEMPORAL_FEATURES: List[str] = field(default_factory=get_temporal_features)
    
    # Date columns
    DATE_COLUMNS: List[str] = field(default_factory=get_date_columns)

    ID_COLUMNS: List[str] = field(default_factory=get_id_columns)
    
    # File paths
    INPUT_FILE: str = 'all_stocks_2006-01-01_to_2018-01-01.csv'
    PROCESSED_FILE: str = 'processed_data.pkl'