from typing import Union

import numpy as np
import torch
import torch.nn.functional as F

from .module import BaseModule
from .layer import FeaturesEmbedding, MultilayerPerceptrons

class NCF(torch.nn.Module):
    """
    Neural Collaborative Filtering
    """
    def __init__(self, field_dims:list, 
                 embed_dim:int = 16,
                 hidden_dims:list = [32, 16, 8], 
                 activations:Union[str, list] = "relu", 
                 dropouts:Union[float, list] = 0.5, 
                 batch_norm:bool = True):
        super().__init__()

        # Embedding layer
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)

        # Multilayer perceptrons
        self.mlp = MultilayerPerceptrons(input_dim = embed_dim * 2, 
                                         hidden_dims = hidden_dims, 
                                         activations = activations, 
                                         dropouts = dropouts,
                                         output_layer = "relu",
                                         batch_norm = batch_norm)

    def forward(self, x):
        # Generate embeddings
        x_embed = self.embedding(x)

        # Concatenate embeddings
        x_embed = torch.flatten(x_embed, start_dim = 1)

        # Prediction
        y = self.mlp(x_embed)

        return y

class NCFModule(BaseModule):
     def __init__(self, field_dims:list, 
                 embed_dim:int = 16,
                 hidden_dims:list = [32, 16, 8], 
                 activations:Union[str, list] = "relu", 
                 dropouts:Union[float, list] = 0.5, 
                 batch_norm:bool = True,
                 lr:float = 1e-3,
                 weight_decay:float = 1e-6,
                 criterion:F = F.mse_loss,
                 save_hyperparameters:bool = True):
        super().__init__(lr, weight_decay, criterion)
        self.model = NCF(field_dims, embed_dim, hidden_dims, activations, dropouts, batch_norm)
        self.save_hyperparameters()
    
        