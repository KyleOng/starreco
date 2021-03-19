from typing import Union

import numpy as np
import torch
import torch.nn.functional as F

from starreco.model import (FeaturesEmbedding, 
                            MultilayerPerceptrons, 
                            Module)

class NMF(Module):
    """
    Neural Matrix Factorization
    """
    def __init__(self, feature_dims, 
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

        :param feature_dims (list): List of feature dimensions. 

        :param embed_dim (int): Embeddings dimensions. Default: 8

        :param hidden_dims (list): List of number of hidden nodes. Default: [32, 16, 18]

        :param activations (str/list): List of activation functions. If type str, then the activation will be repeated len(hidden_dims) times in a list. Default: "relu"

        :param dropouts (float/list): List of dropouts. If type float, then the dropout will be repeated len(hidden_dims) times in a list. Default: 0.5

        :param batch_norm (bool): If True, apply batch normalization on every hidden layer except the combination layer. Default: True

        :param lr (float): Learning rate. Default: 1e-3

        :param weight_decay (float): L2 regularization weight decap: Default: 1e-3

        :param criterion (F): Objective function. Default: F.mse_loss
        """

        super().__init__(lr, weight_decay, criterion)

        # Embedding layer
        self.embedding = FeaturesEmbedding(feature_dims, embed_dim)

        # Neural Collaborative Filtering
        """if type(activations) == str: # Redundant, as this has been taken care in MultilayerPerceptrons()
            activations = np.tile([activations], len(hidden_dims))
        if type(dropouts) == float or type(dropouts) == int:
            dropouts = np.tile([dropouts], len(hidden_dims))"""
        # Number of nodes in the input layers = embed_dim * 2
        self.ncf = MultilayerPerceptrons(embed_dim * 2, 
                                         hidden_dims, 
                                         activations, 
                                         dropouts,
                                         output_layer = None)

        # Combine layer with 1 layer of Multilayer Perceptrons
        self.nmf = MultilayerPerceptrons(hidden_dims[-1] + 1, 
                                         output_layer = "relu")

    def forward(self, x):
        # Generate embeddings
        x = self.embedding(x)
        user_embedding = x[:, 0]
        item_embedding = x[:, 1]

        # Matrix Factorization
        # Dot product between embedding
        matrix = torch.mm(user_embedding, item_embedding.T)
        # Obtain diagonal part of the outer product result
        diagonal = matrix.diag()
        # Reshape
        mf = diagonal.view(diagonal.shape[0], -1)

        # Neural Collaborative Filtering
        # Concatenate embeddings 
        concat = torch.flatten(x, start_dim = 1)
        # NCF prediction
        ncf = self.ncf(concat)

        # Neural Matrix Factorization
        # Concatenate result from MF and NCF
        combine = torch.cat((mf, ncf), 1)
        # Prediction
        return self.nmf(combine)