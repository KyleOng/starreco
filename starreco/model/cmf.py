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
    def __init__(self, users_dim:int, vocab_size:int, max_len:int,
                 user_embed_dim:int = 50, 
                 word_embed_dim:int = 200, 
                 conv_filter_size:int = 100, 
                 conv_kernel_size:int = 3, 
                 conv_activation:str = "relu",
                 fc_hidden_dims:list = [200], 
                 fc_activations:Union[str, list] = "tanh", 
                 fc_dropouts:Union[float, list] = 0.2,
                 batch_norm:bool = True, 
                 lr:float = 1e-3,
                 weight_decay:float = 1e-3,
                 criterion:F = F.mse_loss,
                 save_hyperparameters:bool = True):
        """
        Hyperparameters setting. 

        :param users_dim (int): Number of unique user features.

        :param vocal_size (int): Vocabulary size.

        :param max_len (int): Maximum sentence length.

        :param user_embed_dim (int): User embeddings size. Default: 200

        :param word_embed_dim (int): Word embeddings size. Default: 200

        :param conv_filter_size (int): Convolution filter/depth/channel size. Default: 100

        :param conv_kernel_size (int): Convolution kernel/window size. Default: 3

        :param conv_activation (str): Convolution activation function. Default: "relu".

        :param fc_hidden_dims (list): List of number of hidden nodes. Default: [200]

        :param fc_activations (str/list): List of activation functions. If type str, then the activation will be repeated len(fc_hidden_dims) times in a list. Default: "tanh" 

        :param fc_dropouts (float/list): List of dropouts. If type float, then the dropout will be repeated len(fc_hidden_dims) times in a list. Default: 0.2

        :param batch_norm (bool): If True, apply batch normalization on every hidden layer. Default: True

        :param lr (float): Learning rate. Default: 1e-3

        :param weight_decay (float): L2 regularization weight decay: Default: 1e-3

        :param criterion (F): Objective function. Default: F.mse_loss
        """
        super().__init__(lr, weight_decay, criterion)

        # Embedding layer
        self.user_embedding = FeaturesEmbedding([users_dim], user_embed_dim)  

        # Word embedding layer   
        self.word_embedding = FeaturesEmbedding([vocab_size], word_embed_dim)  

        # CNN
        cnn_blocks = []
        # Convolution layer
        """
        The number of convolution layers (+ pooling) is 1, as CNN is used to address the 
        problem of  varying sentences length by taking the maximum (pooling) of the convoluted 
        sentence embeddings.
        """
        cnn_blocks.append(torch.nn.Conv2d(1, conv_filter_size, 
                                          (conv_kernel_size, word_embed_dim)))
        # Batch normalization
        if batch_norm:
            cnn_blocks.append(torch.nn.BatchNorm2d(conv_filter_size))
        # Activation function
        cnn_blocks.append(ActivationFunction(conv_activation))
        # Max pooling: to address the problem of varying sentence lengths
        cnn_blocks.append(torch.nn.MaxPool2d((max_len - conv_kernel_size + 1, 1)))
        # Flatten
        cnn_blocks.append(torch.nn.Flatten())
        # FC layer
        fc = MultilayerPerceptrons(input_dim = conv_filter_size, 
                                   hidden_dims = [*fc_hidden_dims, user_embed_dim], 
                                   activations = fc_activations, 
                                   dropouts = fc_dropouts,
                                   apply_last_bndp = True, # Check result
                                   output_layer = None, 
                                   batch_norm = batch_norm)
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

        matrix = torch.mm(embed_user, embed_item.T)
        # Obtain diagonal part of matrix as result
        diagonal = matrix.diag()
        return diagonal.view(diagonal.shape[0], -1)