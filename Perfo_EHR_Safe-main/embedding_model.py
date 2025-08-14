# embedding_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional

class CategoricalEmbedding(nn.Module):
    def __init__(
        self,
        num_categories: int,
        embedding_dim: int,
        hidden_dims: List[int],
        dropout_rate: float = 0.2
    ):
        super().__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(num_categories, embedding_dim)
        
        # Create MLP layers for autoencoder
        layers = []
        current_dim = embedding_dim
        
        # Encoder layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            current_dim = hidden_dim
            
        # Decoder layers
        hidden_dims.reverse()
        for hidden_dim in hidden_dims[1:]:
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            current_dim = hidden_dim
            
        # Output layer
        layers.append(nn.Linear(current_dim, num_categories))
        
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get embeddings
        embedded = self.embedding(x)
        
        # Pass through autoencoder
        output = self.layers(embedded)
        
        return output
    
    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Get only the embeddings without passing through the autoencoder"""
        return self.embedding(x)

class EmbeddingTrainer:
    def __init__(
        self,
        model: CategoricalEmbedding,
        learning_rate: float,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        
    def train_step(self, x: torch.Tensor) -> float:
        self.model.train()
        self.optimizer.zero_grad()
        
        # Move data to device
        x = x.to(self.device)
        
        # Forward pass
        output = self.model(x)
        
        # Calculate loss
        loss = self.criterion(output, x)
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def validate(self, x: torch.Tensor) -> float:
        self.model.eval()
        with torch.no_grad():
            x = x.to(self.device)
            output = self.model(x)
            loss = self.criterion(output, x)
        return loss.item()