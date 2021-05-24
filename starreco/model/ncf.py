from typing import Union

import numpy as np
import torch
import torch.nn.functional as F

from .module import BaseModule
from .layer import FeaturesEmbedding, MultilayerPerceptrons

# Done
class NCF(BaseModule):
    """
    Neural Collaborative Filtering.

    - field_dims (list): List of features dimensions.
    - embed_dim (int): Embedding dimension.
    - hidden_dims (list): List of numbers of neurons throughout the hidden layers. Default: [32,16,8].
    - activations (str/list): List of activation functions. Default: relu.
    - dropouts (int/float/list): List of dropout values. Default: 0.5.
    - batch_norm (bool): If True, apply batch normalization in every layer. Batch normalization is applied between activation and dropout layer. Default: True.
    - lr (float): Learning rate.
    - l2_lambda (float): L2 regularization rate.
    - criterion (F): Criterion or objective or loss function.
    """

    def __init__(self, field_dims:list, 
                 embed_dim:int = 8,
                 hidden_dims:list = [32, 16, 8], 
                 activations:Union[str, list] = "relu", 
                 dropouts:Union[int, float, list] = 0.5, 
                 batch_norm:bool = True,
                 lr:float = 1e-3,
                 l2_lambda:float = 1e-3,
                 criterion:F = F.mse_loss):
        super().__init__(lr, l2_lambda, criterion)
        self.save_hyperparameters()
        
        # Embedding layer
        self.features_embedding = FeaturesEmbedding(field_dims, embed_dim)

        # Network 
        self.net = MultilayerPerceptrons(input_dim = embed_dim * 2, 
                                         hidden_dims = hidden_dims, 
                                         activations = activations, 
                                         dropouts = dropouts,
                                         output_layer = "relu",
                                         batch_norm = batch_norm)

    def concatenate(self, x):
        # Generate embeddings
        x_embed = self.features_embedding(x.int())

        # Concatenate embeddings
        return torch.flatten(x_embed, start_dim = 1)

    def forward(self, x):
        concat = self.concatenate(x)
        
        # Feed concatenate embeddings to network
        y = self.net(concat)

        return y
        