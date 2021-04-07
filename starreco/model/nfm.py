from typing import Union

import numpy as np
import torch
import torch.nn.functional as F

from starreco.model import (FeaturesEmbedding, 
                            FeaturesLinear, 
                            PairwiseInteraction, 
                            MultilayerPerceptrons, 
                            Module)

class NFM(Module):
    """
    Neural Factorization Machine
    """
    def __init__(self, field_dims:list, 
                 embed_dim:int = 64, 
                 hidden_dims:list = [1024, 512, 256], 
                 activations:Union[str, list]  = "relu", 
                 dropouts:Union[float, list] = 0.2,
                 batch_norm:bool = True, 
                 lr:float = 1e-3,
                 weight_decay:float = 1e-3,
                 criterion:F = F.mse_loss,
                 save_hyperparameters:bool = True):
        """
        Hyperparameters setting.

        :param field_dims (list): List of field dimensions. 

        :param embed_dim (int): Embeddings dimensions. Default: 8

        :param hidden_dims (list): List of number of hidden nodes. Default: [1024, 512, 256]

        :param activations (str/list): List of activation functions. If type str, then the activation will be repeated len(hidden_dims) times in a list. Default: "relu"

        :param dropouts (float/list): List of dropouts. If type float, then the dropout will be repeated len(hidden_dims) + 1 times in a list and the 1st additional dropout will be applied to b interaction and the rest for multilayer perceptrons (also apply to list type). Default: 0.2

        :param batch_norm (bool): If True, apply batch normalization on every hidden layer. Default: True

        :param lr (float): Learning rate. Default: 1e-3

        :param weight_decay (float): L2 regularization weight decay: Default: 1e-3

        :param criterion (F): Objective function. Default: F.mse_loss
        """
        super().__init__(lr, weight_decay, criterion)

        # Embedding layer
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)

        # Linear layer
        self.linear = FeaturesLinear(field_dims)

        # Bi-interaction layer
        if type(dropouts) == float or type(dropouts) == int:
            dropouts = np.tile([dropouts], len(hidden_dims) + 1)
        self.bi = torch.nn.Sequential(
            PairwiseInteraction(reduce_sum = False),
            torch.nn.BatchNorm1d(embed_dim),
            torch.nn.Dropout(dropouts[0])
        )

        # Multilayer Perceptrons
        self.mlp = MultilayerPerceptrons(input_dim = embed_dim,
                                         hidden_dims = hidden_dims, 
                                         activations = activations, 
                                         dropouts = dropouts[1:],
                                         output_layer = "relu",
                                         batch_norm = batch_norm)
        
    def forward(self, x):
        """
        Perform operations.

        :x (torch.tensor): Input tensors of shape (batch_size, len(field_dims)).

        :return (torch.tensor): Output prediction tensors of shape (batch_size, 1)
        """
        # Generate embeddings
        embed_x = self.embedding(x)

        # Bi-interaction
        cross_term = self.bi(embed_x)

        # Prediction
        return self.linear(x) + self.mlp(cross_term)