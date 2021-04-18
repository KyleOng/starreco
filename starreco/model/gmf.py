from typing import Union

import numpy as np
import torch
import torch.nn.functional as F

from .module import BaseModule
from .layer import FeaturesEmbedding, MultilayerPerceptrons

class GMF(torch.nn.Module):
    """
    Generalized Matrix Factorization
    """
    def __init__(self, field_dims:list, 
                 embed_dim:int = 8):
        super().__init__()

        # Embedding layer
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)

         # Singlelayer perceptrons
        self.slp = MultilayerPerceptrons(input_dim = embed_dim, 
                                         output_layer = "relu")

    def forward(self, x):
        # Generate embeddings
        x_embed = self.embedding(x)
        # Seperate user (1st column) and item (2nd column) embeddings from generated embeddings
        user_embed = x_embed[:, 0]
        item_embed = x_embed[:, 1]

        # Element wise product between user and items embeddings
        product = user_embed * item_embed
        
        # Feed element wise product to generalized non-linear layer
        y = self.slp(product)

        return y

class GMFModule(BaseModule):
     def __init__(self, field_dims:list, 
                  embed_dim:int = 8, 
                  lr:float = 1e-2,
                  weight_decay:float = 1e-6,
                  criterion:F = F.mse_loss):
        super().__init__(lr, weight_decay, criterion)
        self.model = GMF(field_dims, embed_dim)
        self.save_hyperparameters()
    
        