import math

import torch
import torch.nn.functional as F

from starreco.model import (FeaturesEmbedding, 
                            ActivationFunction, 
                            MultilayerPerceptrons, 
                            Module)

class CNNDCF(Module):
    """
    Convolutional Neural Networks based Deep Collaborative Filtering model.
    """
    def __init__(self, field_dims:list, 
                 embed_dim:int = 32, #or 64
                 conv_filter_size:int = 32, #or 64
                 conv_kernel_size:int = 2, 
                 conv_activation:str = "relu", 
                 conv_stride:int = 2,
                 batch_norm:bool = True, 
                 lr:float = 1e-3,
                 weight_decay:float = 1e-3,
                 criterion:F = F.mse_loss,
                 save_hyperparameters:bool = True):
        """
        Hyperparameters setting.      
        
        :param field_dims (list): List of field dimensions.

        :param embed_dim (int): Embedding size. Default: 32 or 64

        :param conv_filter_size (int): Convolution filter/depth/channel size. Default: 32 or 64

        :param conv_kernel_size (int): Convolution kernel/window size. Default: 2

        :param conv_activation (str): Convolution activation function. Default: "relu"

        :param conv_stride (int): Convolution stride size. Default: 2

        :param batch_norm (bool): If True, apply batch normalization on every hidden layer. Default: True

        :param lr (float): Learning rate. Default: 1e-3

        :param weight_decay (float): L2 regularization weight decay: Default: 1e-3

        :param criterion (F): Objective function. Default: F.mse_loss
        """
        super().__init__(lr, weight_decay, criterion)

        # Embedding layer
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)

        # 1 fully connected layer before residual connection
        self.fc_rc = MultilayerPerceptrons(input_dim = embed_dim ** 2,
                                           hidden_dims = [embed_dim ** 2],
                                           apply_last_bndp = False,
                                           output_layer = None)

        # Convolution neural network
        cnn_blocks = [torch.nn.LayerNorm(embed_dim)]
        # The number of convolution = math.ceil(math.log(embed_dim, conv_kernel_size))
        for i in range(math.ceil(math.log(embed_dim, conv_kernel_size))):
            input_channel_size = conv_filter_size if i else 1
            output_channel_size = conv_filter_size
            # Convolution 
            cnn_blocks.append(torch.nn.Conv2d(input_channel_size, 
                                              output_channel_size, 
                                              conv_kernel_size, 
                                              stride = conv_stride))
            # Batch normalization
            if batch_norm:
                cnn_blocks.append(torch.nn.BatchNorm2d(output_channel_size))
            # Activation function
            cnn_blocks.append(ActivationFunction(conv_activation))
        # Flatten
        cnn_blocks.append(torch.nn.Flatten())
        # 1 fully connected layer
        """
        Similar to ONCF, the author specified that there are only 2 layers (input and output layers) in the FC layer, as 1 layer MLP in NCF has more parameters than several layers of convolution in ONCF, which makes it more stable and generalizable than MLP in NCF.
        """
        fc = MultilayerPerceptrons(input_dim = conv_filter_size,
                                   output_layer = "relu")
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