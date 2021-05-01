from typing import Union

import numpy as np
import torch
import torch.nn.functional as F

from .module import BaseModule
from .layer import FeaturesEmbedding, MultilayerPerceptrons

class NCFmodel(torch.nn.Module):
    """
    Neural Collaborative Filtering model
    """

    def __init__(self, field_dims:list, 
                 embed_dim:int,
                 hidden_dims:list, 
                 activations:Union[str, list], 
                 dropouts:Union[int, float, list], 
                 batch_norm:bool):
        super().__init__()

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


class NCF(BaseModule):
    """
    Neural Collaborative Filtering lightning module
    """

    def __init__(self, field_dims:list, 
                 embed_dim:int = 8,
                 hidden_dims:list = [32, 16, 8], 
                 activations:Union[str, list] = "relu", 
                 dropouts:Union[int, float, list] = 0.5, 
                 batch_norm:bool = True,
                 lr:float = 1e-3,
                 weight_decay:float = 1e-6,
                 criterion:F = F.mse_loss,
                 save_hyperparameters:bool = True):
        super().__init__(lr, weight_decay, criterion)
        self.model = NCFmodel(field_dims, embed_dim, hidden_dims, activations, dropouts, batch_norm)
        self.save_hyperparameters()

    def forward(self, x):
        return self.model.forward(x)
        