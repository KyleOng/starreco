import numpy as np
import torch
import torch.nn.functional as F

from starreco.model import (FeaturesEmbedding, 
                            ActivationFunction,
                            MultilayerPerceptrons, 
                            Module)


class CMF(Module):
    def __init__(self, user_features_dim:int, vocab_size:int, max_len:int,
                 user_embed_dim:int = 200, word_embed_dim:int = 200, 
                 conv_filter_size:int = 100, conv_kernel_size:int = 3, conv_activation:str = "relu",
                 fc_hidden_layers:list = [200], fc_activation:str = "tanh", fc_dropouts:list = [0.2],
                 batch_norm:bool = True, criterion:F = F.mse_loss):
        """
        Convolutional Matrix Factorization with explicit parameter settings.

        :param user_features_dim (int): Number of unique user features.

        :param vocal_size (int): Vocabulary size.

        :param max_len (int): Maximum sentence length.

        :user_embed_dim (int): User embeddings size. Default: 200.

        :word_embed_dim (int): Word embeddings size. Default: 200.

        :conv_filter_size (int): Convolution filter/depth/channel size. Default: 100.

        :conv_kernel_size (int): Convolution kernel/window size. Default: 3.

        :conv_activation (str): Name of the activation function for convolution layer. 
        Default: relu.

        :fc_hidden_layers (list): List of number of nodes of the hidden layers (located
        between the input layer and output layer) for the fully connected layers. The 
        number of node for the input and output layer will be automatically defined. 
        The number of input nodes = conv_filter_size
        The number of output nodes = user_embed_dim
        Default: [200]

        :fc_activation (str): Name of the activation function for fully connected layers.
        The number of activation functions will be later repeated based on the length of 
        fc_hidden_layers.
        Default: tanh.

        :fc_dropouts (list): List of dropouts for the fully connected layer.
        The number of dropouts = len(fc_hidden_layers)
        Default: [0.2]
        """
        super().__init__()
        self.criterion = criterion

        # Embedding layer
        self.user_embedding = FeaturesEmbedding([user_features_dim], user_embed_dim)  

        # Word embedding layer   
        self.word_embedding = FeaturesEmbedding([vocab_size], word_embed_dim)  

        # Convolutional neural network
        cnn_blocks = []
        input_channel_size = 1
        output_channel_size = conv_filter_size
        # Convolution layer
        cnn_blocks.append(torch.nn.Conv2d(input_channel_size, output_channel_size, 
                                          (conv_kernel_size, word_embed_dim)))
        # Activation function
        cnn_blocks.append(ActivationFunction(conv_activation))
        # Batch normalization
        if batch_norm:
            cnn_blocks.append(torch.nn.BatchNorm2d(output_channel_size))
        # Max pooling: to address the problem of varying sentence lengths
        cnn_blocks.append(torch.nn.MaxPool2d((max_len - conv_kernel_size + 1, 1)))
        cnn_blocks.append(torch.nn.Flatten())
        # Fully connected layer
        fc_hidden_layers.insert(0, conv_filter_size)
        fc_hidden_layers.append(user_embed_dim)
        fc = MultilayerPerceptrons(fc_hidden_layers, 
                                   np.tile([fc_activation], len(fc_hidden_layers) - 1), 
                                   fc_dropouts, 
                                   batch_norm)
        cnn_blocks.append(fc)
        self.cnn = torch.nn.Sequential(*cnn_blocks)

    def forward(self, x):
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