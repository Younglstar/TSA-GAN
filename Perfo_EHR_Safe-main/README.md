# Perfo_EHR_Safe
Synthetic data generation following the EHR safe framework.

These steps are a recap of all the necessary files and their configurations from preprocessing to final analysis:

1. **Preprocessing Phase**
```python
# config.py - Base configuration
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
    STATIC_FEATURES: List[str] = field(default_factory=get_static_features)
    TEMPORAL_FEATURES: List[str] = field(default_factory=get_temporal_features)
    DATE_COLUMNS: List[str] = field(default_factory=get_date_columns)
    INPUT_FILE: str = 'Emergency_Cohort_with_Followup_500_subject.csv'
    PROCESSED_FILE: str = 'processed_data.pkl'

# data_loader.py, missing_pattern_analyzer.py and main.py
# Output: processed_data.pkl
```

2. **Stochastic Normalization Phase**
```python
# normalizer_config.py
@dataclass
class NormalizerConfig:
    STATIC_NUMERICAL_FEATURES: List[str] = field(
        default_factory=lambda: ['AGE_ARRIVEE', 'GP_ONFILE']
    )
    TEMPORAL_NUMERICAL_FEATURES: List[str] = field(
        default_factory=lambda: ['DUREE_SEJOUR_HOUR']
    )
    NORMALIZATION_PARAMS_FILE: str = 'normalization_params.pkl'

# Output: normalized_data.pkl
```

3. **Categorical Embeddings Phase**
```python
# embedding_config.py
@dataclass
class EmbeddingConfig:
    DEFAULT_EMBEDDING_DIM: int = 8
    MIN_EMBEDDING_DIM: int = 4
    MAX_EMBEDDING_DIM: int = 32
    BATCH_SIZE: int = 64
    LEARNING_RATE: float = 0.001
    NUM_EPOCHS: int = 100
    HIDDEN_LAYERS: List[int] = field(default_factory=lambda: [64, 32])
    DROPOUT_RATE: float = 0.2
    EMBEDDING_MODELS_FILE: str = 'embedding_models.pkl'
    EMBEDDINGS_FILE: str = 'categorical_embeddings.pkl'

# Output: categorical_embeddings.pkl
```

4. **Encoder-Decoder Phase**
```python
# encdec_config.py
@dataclass
class EncoderDecoderConfig:
    ENCODER_HIDDEN_DIMS: List[int] = field(
        default_factory=lambda: [512, 256, 128]
    )
    DECODER_HIDDEN_DIMS: List[int] = field(
        default_factory=lambda: [128, 256, 512]
    )
    LATENT_DIM: int = 64
    LEARNING_RATE: float = 0.001
    BATCH_SIZE: int = 32
    NUM_EPOCHS: int = 100
    DROPOUT_RATE: float = 0.2
    USE_BATCH_NORM: bool = True
    MODEL_SAVE_PATH: str = 'encoder_decoder_model.pkl'
    ENCODED_DATA_PATH: str = 'encoded_data.pkl'

# Output: encoded_data.pkl, feature_dims.pkl
```

5. **GAN Phase**
```python
# gan_config.py
@dataclass
class GANConfig:
    NOISE_DIM: int = 128
    GENERATOR_HIDDEN_DIMS: List[int] = field(
        default_factory=lambda: [256, 512, 1024]
    )
    DISCRIMINATOR_HIDDEN_DIMS: List[int] = field(
        default_factory=lambda: [1024, 512, 256]
    )
    LEARNING_RATE_G: float = 0.0002
    LEARNING_RATE_D: float = 0.0002
    BATCH_SIZE: int = 64
    NUM_EPOCHS: int = 500
    N_CRITIC: int = 5
    GRAD_PENALTY_WEIGHT: float = 10.0
    DROPOUT_RATE: float = 0.2
    USE_BATCH_NORM: bool = True
    MODEL_SAVE_PATH: str = 'gan_model.pkl'
    SYNTHETIC_DATA_PATH: str = 'synthetic_data.pkl'
    NUM_SYNTHETIC_SAMPLES: int = 1000

# Output: synthetic_data.pkl, scaling_params.pkl
```

6. **Analysis Phase**
```python
# Directory structure for results
comparison_plots/
    ├── feature_distributions.png
    ├── statistical_moments.png
    ├── correlation_analysis.png
    ├── correlation_preservation.png
    ├── density_comparison.png
    ├── summary_report.txt
    └── summary_statistics.pkl
```

Complete Pipeline Flow:
```
Raw Data (CSV) 
  → Preprocessing (processed_data.pkl)
  → Normalization (normalized_data.pkl)
  → Embeddings (categorical_embeddings.pkl)
  → Encoder-Decoder (encoded_data.pkl)
  → GAN Training (synthetic_data.pkl)
  → Analysis (comparison_plots/)
```

Required Libraries:
```bash
pip install torch numpy pandas matplotlib seaborn scikit-learn scipy tqdm
```

Running the Pipeline:
```bash
# 1. Preprocessing
python main.py

# 2. Normalization
python normalize_main.py

# 3. Embeddings
python embedding_main.py

# 4. Encoder-Decoder
python train_encdec.py

# 5. GAN
python train_gan.py

# 6. Analysis
python enhanced_comparison.py
```

File Dependencies:
```
Data Files Chain:
CSV → processed_data.pkl → normalized_data.pkl → 
categorical_embeddings.pkl → encoded_data.pkl → synthetic_data.pkl

Model Files:
- embedding_models.pkl
- encoder_decoder_model.pkl
- gan_model.pkl
- scaling_params.pkl
- feature_dims.pkl
```

This recap covers all the major components needed to run the complete EHR-Safe pipeline from raw data to synthetic data generation and analysis.
