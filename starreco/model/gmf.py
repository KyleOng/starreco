from typing import Union

import numpy as np
import torch
import torch.nn.functional as F

from .module import BaseModule
from .layers import FeaturesEmbedding, MultilayerPerceptrons
from .mixins import FeaturesEmbeddingMixin

# Done
class GMF(BaseModule, FeaturesEmbeddingMixin):
    """
    Generalized Matrix Factorization.

    - field_dims (list): List of features dimensions.
    - embed_dim (int): Embedding dimension. Default: 8.
    - lr (float): Learning rate. Default: 1e-3.
    - l2_lambda (float): L2 regularization rate. Default: 1e-3.
    - criterion (F): Criterion or objective or loss function. Default: F.mse_loss.
    """

    def __init__(self, 
                 field_dims:list, 
                 embed_dim:int = 8, 
                 lr:float = 1e-3,
                 l2_lambda:float = 1e-6,
                 criterion:F = F.mse_loss):
        super().__init__(lr, l2_lambda, criterion)
        self.save_hyperparameters()

        # Embedding layer
        self.features_embedding = FeaturesEmbedding(field_dims, embed_dim)

        # Network
        self.net = MultilayerPerceptrons(input_dim = embed_dim, 
                                         output_layer = "relu")

    def forward(self, x):
        # Generate embeddings
        user_embed, item_embed = self.user_item_embeddings(x)

        # Element wise product between user and items embeddings
        product = user_embed * item_embed
        
        # Prediction
        y = self.net(product)

        return y
        