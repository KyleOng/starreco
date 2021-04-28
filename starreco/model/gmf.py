from typing import Union

import numpy as np
import torch
import torch.nn.functional as F

from .module import BaseModule
from .layer import FeaturesEmbedding, MultilayerPerceptrons

class GMF(BaseModule):
    """
    Matrix Factorization
    """

    def __init__(self, field_dims:list, 
                 embed_dim:int = 8, 
                 lr:float = 1e-3,
                 weight_decay:float = 1e-6,
                 criterion:F = F.mse_loss):
        super().__init__(lr, weight_decay, criterion)

        # Embedding layer
        self.features_embedding = FeaturesEmbedding(field_dims, embed_dim)

        # Network
        self.net = MultilayerPerceptrons(input_dim = embed_dim, 
                                         output_layer = "relu")

        self.save_hyperparameters()

    def forward(self, x):
        # Generate embeddings
        x_embed = self.features_embedding(x.int())
        # Seperate user (1st column) and item (2nd column) embeddings from generated embeddings
        user_embed = x_embed[:, 0]
        item_embed = x_embed[:, 1]

        # Element wise product between user and items embeddings
        product = user_embed * item_embed
        
        # Feed element wise product to generalized non-linear layer
        y = self.net(product)

        return y
    
        