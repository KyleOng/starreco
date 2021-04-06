from typing import Union

import numpy as np
import torch
import torch.nn.functional as F

from starreco.model import (FeaturesEmbedding, 
                            FeaturesLinear, 
                            PairwiseInteraction, 
                            MultilayerPerceptrons, 
                            Module)

class DFM(Module):
    """
    Deep Factorization Machine.
    """
    def __init__(self, feature_dims:list, 
                 embed_dim:int = 10, 
                 hidden_dims:list = [400,400,400], 
                 activations:Union[str, list] = "relu", 
                 dropouts:Union[float, list] = 0.5,
                 batch_norm:bool = True,
                 lr:float = 1e-3,
                 weight_decay:float = 1e-3,
                 criterion:F = F.mse_loss,
                 save_hyperparameters:bool = True):
        """
        Hyperparameters setting.

        :param feature_dims (list): List of feature dimensions.

        :param embed_dim (int): Embedding size. Default 10

        :param hidden_dims (list): List of number of hidden nodes. Default: [400, 400, 400]

        :param activations (str/list): List of activation functions. If type str, then the activation will be repeated len(hidden_dims) times in a list. Default: "relu"

        :param dropouts (float/list): List of dropouts. If type float, then the dropout will be repeated len(hidden_dims) times in a list. Default: 0.5

        :param batch_norm (bool): If True, apply batch normalization on every hidden layer. Default: True

        :param lr (float): Learning rate. Default: 1e-3

        :param weight_decay (float): L2 regularization weight decay: Default: 1e-3

        :param criterion (F): Objective function. Default: F.mse_loss
        """
        super().__init__(lr, weight_decay, criterion)

        # Embedding layer
        self.embedding = FeaturesEmbedding(feature_dims, embed_dim)

        # Linear layer 
        self.linear = FeaturesLinear(feature_dims)

        # Pairwise interaction
        self.pwin = PairwiseInteraction()

        # Multilayer perceptrons
        
        self.mlp = MultilayerPerceptrons(input_dim = len(feature_dims) * embed_dim,
                                         hidden_dims = hidden_dims, 
                                         activations = activations, 
                                         dropouts = dropouts,
                                         output_layer = "relu",
                                         batch_norm = batch_norm)
        
    def forward(self, x):
        """
        Perform operations.

        :x (torch.tensor): Input tensors of shape (batch_size, len(feature_dims))

        :return (torch.tensor): Output prediction tensors of shape (batch_size, 1)
        """
        # Generate embeddings
        embed_x = self.embedding(x)

        # Prediction
        return self.linear(x) + self.pwin(embed_x) + \
        self.mlp(torch.flatten(embed_x, start_dim = 1)) 