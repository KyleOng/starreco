from typing import Union

import numpy as np
import torch
import torch.nn.functional as F

from .module import BaseModule
from .layer import FeaturesEmbedding, MultilayerPerceptrons

class NCF(BaseModule):
    """
    Neural Collaborative Filtering
    """

    def __init__(self, field_dims:list, 
                 embed_dim:int = 8,
                 hidden_dims:list = [32, 16, 8], 
                 activations:Union[str, list] = "relu", 
                 dropouts:Union[float, list] = 0.5, 
                 batch_norm:bool = True,
                 lr:float = 1e-3,
                 weight_decay:float = 1e-6,
                 criterion:F = F.mse_loss,
                 save_hyperparameters:bool = True):
        super().__init__(lr, weight_decay, criterion)

        # Embedding layer
        self.features_embedding = FeaturesEmbedding(field_dims, embed_dim)

        # Network 
        self.net = MultilayerPerceptrons(input_dim = embed_dim * 2, 
                                         hidden_dims = hidden_dims, 
                                         activations = activations, 
                                         dropouts = dropouts,
                                         output_layer = "relu",
                                         batch_norm = batch_norm)

        self.save_hyperparameters()

    def forward(self, x):
        # Generate embeddings
        x_embed = self.features_embedding(x.int())

        # Concatenate embeddings
        concat = torch.flatten(x_embed, start_dim = 1)
        
        # Feed concatenate embeddings to network
        y = self.net(concat)

        return y
