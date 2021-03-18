from typing import Union

import numpy as np
import torch
import torch.nn.functional as F

from starreco.model import (FeaturesEmbedding, 
                            MultilayerPerceptrons, 
                            Module)

class NCF(Module):
    """
    Neural Collaborative Filtering
    """
    def __init__(self, features_dim:list, 
                 embed_dim:int = 8,
                 hidden_dims:list = [32, 16, 8], 
                 activations:Union[str, list] = "relu", 
                 dropouts:Union[float, list] = 0.5, 
                 batch_norm:bool = True,
                 lr:float = 1e-3,
                 weight_decay:float = 1e-3,
                 criterion:F = F.mse_loss):
        """
        Hyperparamters setting.

        :param features_dim (list): List of feature dimensions. 

        :param embed_dim (int): Embeddings dimensions. Default: 8

        :param hidden_dims (list): List of number of hidden nodes. Default: [32, 16, 18]

        :param activations (str/list): List of activation functions. If type str, then the activation will be repeated len(hidden_dims) times in a list. Default: "relu"

        :param dropouts (float/list): List of dropouts. If type float, then the dropout will be repeated len(hidden_dims) times in a list. Default: 0.5

        :batch_norm (bool): If True, apply batch normalization on every hidden layer. Default: True

        :param lr (float): Learning rate. Default: 1e-3

        :param weight_decay (float): L2 regularization weight decap: Default: 1e-3

        :param criterion (F): Objective function. Default: F.mse_loss
        """
        super().__init__(lr, weight_decay, criterion)

        # Embedding layer
        self.embedding = FeaturesEmbedding(features_dim, embed_dim)

        # Multilayer perceptrons
        """if type(activations) == str: # Redundant, as this has been taken care in MultilayerPerceptrons()
            activations = np.tile([activations], len(hidden_dims))
        if type(dropouts) == float:
            dropouts = np.tile([dropouts], len(hidden_dims))"""
        # Number of nodes in the input layers = embed_dim * 2
        self.mlp = MultilayerPerceptrons(embed_dim * 2, 
                                         hidden_dims, 
                                         activations, 
                                         dropouts,
                                         output_layer = "relu")

    def forward(self, x):
        """
        Perform operations.

        :x (torch.tensor): Input tensors of shape (batch_size, len(features_dim)).

        :return (torch.tensor): Output prediction tensors of shape (batch_size, 1)
        """
        # Generate embeddings
        x = self.embedding(x)

        # Concatenate embeddings
        x = torch.flatten(x, start_dim = 1)

        # Prediction
        return self.mlp(x)