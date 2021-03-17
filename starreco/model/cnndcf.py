import math

import torch
import torch.nn.functional as F

from starreco.model import FeaturesEmbedding, ActivationFunction, MultilayerPerceptrons, Module

class CNNDCF(Module):
    """
    ConvolutionalNeural Networks based Deep Collaborative Filtering model
    """
    def __init__(self, features_dim:list, 
                 embed_dim:int = 32, #or 64
                 conv_filter_size:int = 32, #or 64
                 conv_kernel_size:int = 2, 
                 conv_activation:str = "relu", 
                 conv_stride:int = 2,
                 fc_activation:str = "relu",
                 batch_norm:bool = True, 
                 criterion:F = F.mse_loss):
        """
        Explicit parameter settings. 
        
        Similar to ONCF, the author specified that there is only 2 layers (input and output 
        layers) in the FC layer, as 1 layer MLP in NCF has more parameters than several 
        layers of convolution in ONCF, which makes it more stable and generalizable than MLP 
        in NCF.

        :param features_dim (list): List of feature dimensions.

        :param embed_dim (int): Embedding size. Default = 32 or 64

        :param conv_filter_size (int): Convolution filter/depth/channel size. Default = 32

        :param conv_kernel_size (int): Convolution kernel/window size. Default = 2

        :param conv_activation (str): Name of the activation function for convolution layers
        All convolution layer will use the same activation function. Default = 2

        :param fc_activation (str): Activation function name for the FC layer. Default = relu

        :param batch_norm (bool): Batch normalization on CNN, placed after the last pooling layer
        and before FC. Default: True

        :param criterin (F): Objective function. Default = F.mse_lose
        """
        super().__init__()
        self.criterion = criterion

        # Embedding layer
        self.embedding = FeaturesEmbedding(features_dim, embed_dim)

        # 1 fully connected layer before residual connection
        self.fc_rc = MultilayerPerceptrons([embed_dim ** 2, embed_dim ** 2], 
                                          [fc_activation], 
                                          [])

        # Convolution neural network
        cnn_blocks = [torch.nn.LayerNorm(embed_dim)]
        for i in range(math.ceil(math.log(embed_dim, conv_kernel_size))):
            input_channel_size = conv_filter_size if i else 1
            output_channel_size = conv_filter_size
            # Convolution 
            cnn_blocks.append(torch.nn.Conv2d(input_channel_size, 
                                              output_channel_size, 
                                              conv_kernel_size, 
                                              stride = 2))
            # Activation function
            cnn_blocks.append(ActivationFunction(conv_activation))
        # Batch normalization
        if batch_norm:
            cnn_blocks.append(torch.nn.BatchNorm2d(output_channel_size))
        # Flatten
        cnn_blocks.append(torch.nn.Flatten())
        # 1 fully connected layer
        fc = MultilayerPerceptrons([conv_filter_size,1], [fc_activation], [])
        cnn_blocks.append(fc)
        self.cnn = torch.nn.Sequential(*cnn_blocks)

    def forward(self, x):
        """
        Perform operations.

        :x (torch.tensor): Input tensors of shape (batch_size, 2)

        :return (torch.tensor): Output prediction tensors of shape (batch_size, 1)
        """
        # Generate embeddings
        x = self.embedding(x)
        user_embedding = x[:, 0]
        item_embedding = x[:, 1]

        # Outer products on embeddings
        residual = torch.bmm(user_embedding.unsqueeze(2), item_embedding.unsqueeze(1))

        # Fully connected layer before residual connection
        flat = torch.flatten(residual, start_dim = 1)
        out = self.fc_rc(flat).view(residual.shape)

        # Residual connection
        out += residual

        # Prediction
        out = torch.unsqueeze(residual, 1)
        return self.cnn(out)