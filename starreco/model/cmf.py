from typing import Union

import numpy as np
import torch
import torch.nn.functional as F

from starreco.model import (FeaturesEmbedding, 
                            ActivationFunction,
                            MultilayerPerceptrons, 
                            Module)


class CMF(Module):
    """
    Convolutional Matrix Factorization.
    """
    def __init__(self, user_features_dim:int, vocab_size:int, max_len:int,
                 user_embed_dim:int = 50, 
                 word_embed_dim:int = 200, 
                 conv_filter_size:int = 100, 
                 conv_kernel_size:int = 3, 
                 conv_activation:str = "relu",
                 batch_norm:bool = True, 
                 fc_hidden_layers:list = [200], 
                 fc_activations:Union[str, list] = "tanh", 
                 fc_dropouts:Union[float, list] = 0.2,
                 criterion:F = F.mse_loss):
        """
        Explicit parameter settings. 
        
        The number of convolution layers (+ pooling) is 1, as CNN is used to address the 
        problem of  varying sentences length by taking the maximum (pooling) of the convoluted 
        sentence embeddings.

        :param user_features_dim (int): Number of unique user features.

        :param vocal_size (int): Vocabulary size.

        :param max_len (int): Maximum sentence length.

        :param user_embed_dim (int): User embeddings size. Default: 200

        :param word_embed_dim (int): Word embeddings size. Default: 200

        :param conv_filter_size (int): Convolution filter/depth/channel size. Default: 100

        :param conv_kernel_size (int): Convolution kernel/window size. Default: 3

        :param conv_activation (str): Name of the activation function for convolution layer. 
        Default: relu.

        :param fc_hidden_layers (list): List of number of nodes of the hidden layers (located
        between the input layer and output layer) for the FC layers. The number of node for 
        the input and output layer will be automatically defined. 
        Number of input nodes = conv_filter_size. convolution
        Number of output nodes = user_embed_dim
        Default: [200]

        :param fc_activation (str/list): List of activation function names for the FC layers.
        If type int, then the activation function names will be repeated based on 
        the number of fc_hidden_layers.
        Number of activations = 2 + len(fc_hidden_layers) - 1
        Default: tanh 

        :param fc_dropouts (float/list): List of dropouts for the FC layers. If type
        str, then the dropouts will be repeated based on the number of fc_hidden_layers.
        Number of dropouts = 2 + len(fc_hidden_layers) - 2
        Default: 0.2

        :param batch_normm (bool): Batch normalization on CNN, placed after the last pooling layer
        and before FC. Default: True

        :param criterion (F): Objective function. Default: F.mse_loss
        """
        # Make a copy for mutable object to solve the problem of "Mutable Default Argument".
        # For more info: https://stackoverflow.com/a/13518071/9837978
        fc_hidden_layers = fc_hidden_layers.copy()
        super().__init__()
        self.criterion = criterion

        # Embedding layer
        self.user_embedding = FeaturesEmbedding([user_features_dim], user_embed_dim)  

        # Word embedding layer   
        self.word_embedding = FeaturesEmbedding([vocab_size], word_embed_dim)  

        # CNN
        cnn_blocks = []
        input_channel_size = 1
        output_channel_size = conv_filter_size
        # Convolution layer
        cnn_blocks.append(torch.nn.Conv2d(input_channel_size, output_channel_size, 
                                          (conv_kernel_size, word_embed_dim)))
        # Activation function
        cnn_blocks.append(ActivationFunction(conv_activation))
        # Max pooling: to address the problem of varying sentence lengths
        cnn_blocks.append(torch.nn.MaxPool2d((max_len - conv_kernel_size + 1, 1)))
        # Batch normalization
        if batch_norm:
            cnn_blocks.append(torch.nn.BatchNorm2d(output_channel_size))
        # Flatten
        cnn_blocks.append(torch.nn.Flatten())
        # FC layer
        fc_hidden_layers.insert(0, conv_filter_size)
        fc_hidden_layers.append(user_embed_dim)
        if type(fc_activations) == str:
            fc_activations = np.tile([fc_activations], len(fc_hidden_layers) - 1)
        if type(fc_dropouts) == float:
            fc_dropouts = np.tile([fc_dropouts], len(fc_hidden_layers) -2)
        fc = MultilayerPerceptrons(fc_hidden_layers, fc_activations, fc_dropouts)
        cnn_blocks.append(fc)
        self.cnn = torch.nn.Sequential(*cnn_blocks)

    def forward(self, x):
        """
        Perform operations.

        :x (torch.tensor): Input tensors of shape (batch_size, 1 + max_len)

        :return (torch.tensor): Output prediction tensors of shape (batch_size, 1)
        """
        # Generate user embeddings from 1st layer
        embed_user = self.user_embedding(x[:, 0])
        embed_user = embed_user.squeeze(0)

        # Generate word embeddings
        embed_word = self.word_embedding(x[:, 1:])
        embed_word = embed_word.unsqueeze(1)

        # Generate item features aka "embed_item" from word embeddings
        embed_item = self.cnn(embed_word)

        # Matrix factorization, instead of Probabilistic MF
        matrix = torch.mm(embed_user, embed_item.T)
        # Obtain diagonal part of matrix as result
        diagonal = matrix.diag()
        return diagonal.view(diagonal.shape[0], -1)