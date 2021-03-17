from typing import Union

import numpy as np
import torch
import torch.nn.functional as F

from starreco.model import FeaturesEmbedding, MultilayerPerceptrons, Module

class NCF(Module):
    """
    Neural Collaborative Filtering
    """
    def __init__(self, features_dim:list, 
                 embed_dim:int = 8,
                 output_layers:list = [32, 16, 8, 1], 
                 activations:Union[str, list] = "relu", 
                 dropouts:Union[float, list] = 0.5, 
                 criterion:F = F.mse_loss):
        """
         Explicit parameter settings.

        :param features_dim (list): List of feature dimensions. 

        :param embed_dim (int): Embeddings dimensions. Default = 8

        :output_layers (list): List of number of nodes of the hidden and output layers 
        for the fully connected layers. The number of node for the input layer will be 
        automatically defined. 
        Number of input nodes = embed_dim * 2
        Default: [32, 16, 8, 1]

        :fc_activation (str/list): List of activation function names for the fully connect
        layers. If type int, then the activation function names will be repeated based on 
        the number of output_layers.
        Number of activations = 1 + len(output_layers) - 1.
        Default: relu

        :fc_dropouts (float/list): List of dropouts for the fully connected layer. If type
        str, then the dropouts will be repeated based on the number of output_layers.
        Number of dropouts = 1 + len(output_layers) - 2.
        Default: 0.5

        :criterion (F): Objective function. Default: F.mse_loss
        """
        super().__init__()
        self.criterion = criterion

        # Embedding layer
        self.embedding = FeaturesEmbedding(features_dim, embed_dim)

        # Multilayer perceptrons
        output_layers.insert(0, embed_dim * 2)
        if type(activations) == str:
            activations = np.tile([activations], len(output_layers) - 1)
        if type(dropouts) == float:
            dropouts = np.tile([dropouts], len(output_layers) - 2)
        self.mlp = MultilayerPerceptrons(output_layers, activations, dropouts)

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