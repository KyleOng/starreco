from typing import Union

import numpy as np
import torch
import torch.nn.functional as F

from starreco.model import FeaturesEmbedding, FeaturesLinear, PairwiseInteraction, MultilayerPerceptrons, Module

class DFM(Module):
    def __init__(self, features_dim: list, 
                 embed_dim:int = 10, 
                 output_layers:list = [400,400,400,1], 
                 activations:Union[str, list] = "relu", 
                 dropouts:Union[float, list] = 0.5,
                 criterion = F.mse_loss):
        """
        Explicit parameter settings.

        :param features_dim (list): List of feature dimensions.

        :param embed_dim (int): Embedding size. Default 10

        :output_layers (list): List of hidden and output layers. The number of node for
        the input will be automatically defined.
        Number of input nodes = len(features_dim) * embed_dim
        Default = [400, 400, 400, 1]

        :param activation (str/list): List of activation function names. If type int, 
        then the activation function names will be repeated based on the number of 
        output_layers.
        Number of activations = 1 + len(output_layers) -1
        Default: tanh 

        :param dropouts (float/list): List of dropouts. If type str, then the dropouts 
        will be repeated based on the number of output_layers.
        Number of dropouts = 1 + len(output_layers) - 2
        Default: 0.2

        :param criterion (F): Objective function. Default: F.mse_loss
        """
        # Make a copy for mutable object to solve the problem of "Mutable Default Argument".
        # For more info: https://stackoverflow.com/a/13518071/9837978
        output_layers = output_layers.copy()
        super().__init__()
        self.criterion = criterion

        # Embedding layer
        self.embedding = FeaturesEmbedding(features_dim, embed_dim)

        # Linear layer 
        self.linear = FeaturesLinear(features_dim)

        # Pairwise interaction
        self.pwin = PairwiseInteraction()

        # Multilayer perceptrons
        output_layers.insert(0, len(features_dim) * embed_dim)
        if type(activations) == str:
            activations = np.tile([activations], len(output_layers) - 1)
        if type(dropouts) == float:
            dropouts = np.tile([dropouts], len(output_layers) -2)
        self.mlp = MultilayerPerceptrons(output_layers, activations, dropouts)
        
    def forward(self, x):
        """
        Perform operations.

        :x (torch.tensor): Input tensors of shape (batch_size, len(features_dim))

        :return (torch.tensor): Output prediction tensors of shape (batch_size, 1)
        """
        # Generate embeddings
        embed_x = self.embedding(x)

        # Prediction
        return self.linear(x) + self.pwin(embed_x) + \
        self.mlp(torch.flatten(embed_x, start_dim = 1)) 