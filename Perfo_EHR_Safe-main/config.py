# config.py
from dataclasses import dataclass, field
from typing import List

def get_static_features() -> List[str]:
    return [
        'DOSSIER_HASH', 'SEXE', 'MARITAL_STATUS', 
        'GP_ONFILE', 'AGE_ARRIVEE'
    ]

def get_temporal_features() -> List[str]:
    return [
        'ETABLISSEMENT', 'UNITE', 'Diagnostic',
        'RAISON_VISITE', 'DUREE_SEJOUR_HOUR'
    ]

def get_date_columns() -> List[str]:
    return ['YM_ADM', 'YM_ADM_Fol_Urg']

@dataclass
class DataConfig:
    # Static features configuration
    STATIC_FEATURES: List[str] = field(default_factory=get_static_features)
    
    # Temporal features configuration
    TEMPORAL_FEATURES: List[str] = field(default_factory=get_temporal_features)
    
    # Date columns
    DATE_COLUMNS: List[str] = field(default_factory=get_date_columns)
    
    # File paths
    INPUT_FILE: str = 'Emergency_Cohort_with_Followup_500_subject.csv'
    PROCESSED_FILE: str = 'processed_data.pkl'