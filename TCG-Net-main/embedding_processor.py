# embedding_processor.py
import pandas as pd
import numpy as np
import torch
from typing import Dict, Tuple, List
from sklearn.preprocessing import LabelEncoder
import pickle
from embedding_config import EmbeddingConfig
from embedding_model import CategoricalEmbedding, EmbeddingTrainer

class CategoricalEmbeddingProcessor:
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.label_encoders = {}
        self.embedding_models = {}
        self.embedding_trainers = {}
        
    def _collect_all_categories(self, temporal_data: Dict[str, pd.DataFrame], feature: str) -> List[str]:
        """Collect all unique categories for a feature across all visit types"""
        all_categories = set()
        
        # Add 'MISSING' as a possible category
        all_categories.add('MISSING')
        
        # Collect categories from each visit type
        for visit_type, data in temporal_data.items():
            if feature in data.columns:
                categories = data[feature].fillna('MISSING').astype(str).unique()
                all_categories.update(categories)
                
        return sorted(list(all_categories))
    
    def _create_label_encoders_temporal(self, temporal_data: Dict[str, pd.DataFrame], features: List[str]) -> None:
        """Create label encoders for temporal features ensuring all categories are captured"""
        for feature in features:
            # Collect all possible categories across visit types
            all_categories = self._collect_all_categories(temporal_data, feature)
            
            # Create label encoder with all possible categories
            le = LabelEncoder()
            le.fit(all_categories)
            
            self.label_encoders[feature] = le
            self.config.category_mappings[feature] = {
                'categories': le.classes_.tolist(),
                'num_categories': len(le.classes_)
            }
            
            print(f"Created label encoder for {feature} with {len(all_categories)} categories")
    
    def _create_label_encoders_static(self, data: pd.DataFrame, features: List[str]) -> None:
        """Create label encoders for static features"""
        for feature in features:
            if feature in data.columns:
                # Include MISSING in possible categories
                feature_data = data[feature].fillna('MISSING').astype(str)
                unique_values = np.unique(feature_data)
                
                le = LabelEncoder()
                le.fit(unique_values)
                
                self.label_encoders[feature] = le
                self.config.category_mappings[feature] = {
                    'categories': le.classes_.tolist(),
                    'num_categories': len(le.classes_)
                }
                
                print(f"Created label encoder for {feature} with {len(unique_values)} categories")
    
    def _get_embedding_dim(self, num_categories: int) -> int:
        """Determine embedding dimension based on number of categories"""
        dim = min(
            self.config.MAX_EMBEDDING_DIM,
            max(
                self.config.MIN_EMBEDDING_DIM,
                int(np.ceil(num_categories ** 0.25))  # Fourth root rule
            )
        )
        return dim
    
    def _create_embedding_model(self, feature: str) -> None:
        """Create embedding model for a feature"""
        num_categories = self.config.category_mappings[feature]['num_categories']
        embedding_dim = self._get_embedding_dim(num_categories)
        
        model = CategoricalEmbedding(
            num_categories=num_categories,
            embedding_dim=embedding_dim,
            hidden_dims=self.config.HIDDEN_LAYERS,
            dropout_rate=self.config.DROPOUT_RATE
        )
        
        trainer = EmbeddingTrainer(
            model=model,
            learning_rate=self.config.LEARNING_RATE
        )
        
        self.embedding_models[feature] = model
        self.embedding_trainers[feature] = trainer
    
    def fit_transform_static_features(self, static_data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Fit and transform static categorical features"""
        print("Processing static categorical features...")
        self._create_label_encoders_static(static_data, self.config.STATIC_CATEGORICAL_FEATURES)
        
        static_embeddings = {}
        for feature in self.config.STATIC_CATEGORICAL_FEATURES:
            if feature in static_data.columns:
                print(f"\nTraining embeddings for {feature}...")
                
                # Create and train embedding model
                self._create_embedding_model(feature)
                
                # Prepare data
                feature_data = static_data[feature].fillna('MISSING').astype(str)
                encoded_data = self.label_encoders[feature].transform(feature_data)
                tensor_data = torch.LongTensor(encoded_data)
                
                # Train model
                trainer = self.embedding_trainers[feature]
                model = self.embedding_models[feature]
                
                for epoch in range(self.config.NUM_EPOCHS):
                    loss = trainer.train_step(tensor_data)
                    if (epoch + 1) % 20 == 0:
                        print(f"Epoch {epoch+1}/{self.config.NUM_EPOCHS}, Loss: {loss:.4f}")
                
                # Get embeddings
                with torch.no_grad():
                    embeddings = model.get_embeddings(tensor_data).cpu().numpy()
                static_embeddings[feature] = embeddings
        
        return static_embeddings
    
    def fit_transform_temporal_features(
        self, 
        temporal_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """Fit and transform temporal categorical features"""
        print("\nProcessing temporal categorical features...")
        
        # First, create label encoders using all temporal data
        print("Creating label encoders for temporal features...")
        self._create_label_encoders_temporal(temporal_data, self.config.TEMPORAL_CATEGORICAL_FEATURES)
        
        temporal_embeddings = {}
        
        # Process each visit type
        for visit_type, data in temporal_data.items():
            print(f"\nProcessing {visit_type} visit data...")
            temporal_embeddings[visit_type] = {}
            
            # Process each feature
            for feature in self.config.TEMPORAL_CATEGORICAL_FEATURES:
                if feature in data.columns:
                    print(f"\nTraining embeddings for {feature}...")
                    
                    # Create embedding model if not exists
                    if feature not in self.embedding_models:
                        self._create_embedding_model(feature)
                    
                    # Prepare data
                    feature_data = data[feature].fillna('MISSING').astype(str)
                    encoded_data = self.label_encoders[feature].transform(feature_data)
                    tensor_data = torch.LongTensor(encoded_data)
                    
                    # Train model
                    trainer = self.embedding_trainers[feature]
                    model = self.embedding_models[feature]
                    
                    for epoch in range(self.config.NUM_EPOCHS):
                        loss = trainer.train_step(tensor_data)
                        if (epoch + 1) % 20 == 0:
                            print(f"Epoch {epoch+1}/{self.config.NUM_EPOCHS}, Loss: {loss:.4f}")
                    
                    # Get embeddings
                    with torch.no_grad():
                        embeddings = model.get_embeddings(tensor_data).cpu().numpy()
                    temporal_embeddings[visit_type][feature] = embeddings
        
        return temporal_embeddings
    
    def save_models(self) -> None:
        """Save embedding models and label encoders"""
        save_dict = {
            'label_encoders': self.label_encoders,
            'embedding_models': self.embedding_models,
            'category_mappings': self.config.category_mappings
        }
        with open(self.config.EMBEDDING_MODELS_FILE, 'wb') as f:
            pickle.dump(save_dict, f)
    
    def load_models(self) -> None:
        """Load embedding models and label encoders"""
        with open(self.config.EMBEDDING_MODELS_FILE, 'rb') as f:
            save_dict = pickle.load(f)
        self.label_encoders = save_dict['label_encoders']
        self.embedding_models = save_dict['embedding_models']
        self.config.category_mappings = save_dict['category_mappings']