import math
from typing import Union
from functools import reduce

import torch
import torch.nn.functional as F

from starreco.model import (MultilayerPerceptrons,
                            Module)

class CCAE(Module):
    """
    Collaborative Convolution Autoencoder
    """
    def __init__(self, io_dim:int,
                 conv_filter_sizes:list = [16, 8], 
                 conv_kernel_size:int = 3, 
                 conv_pooling_size:int = 2, 
                 e_conv_activations:str = "tanh", 
                 d_conv_activations:str = "tanh", 
                 dense_refeeding:int = 1,
                 batch_norm:bool = True,
                 lr:float = 1e-3,
                 weight_decay:float = 1e-3,
                 criterion:F = F.mse_loss):
        """
        Hyperparameters setting.

        :param io_dim (int): Input/Output dimension.

        :param dropout (float): Dropout within latent space. Default: 0.5

        :param dense_refeeding (int): Number of dense refeeding: Default: 1

        :param batch_norm (bool): If True, apply batch normalization on every hidden layer. Default: True

        :param lr (float): Learning rate. Default: 1e-3

        :param weight_decay (float): L2 regularization weight decay: Default: 1e-3

        :param criterion (F): Objective function. Default: F.mse_loss
        """
        super().__init__(lr, weight_decay, criterion)
        self.dense_refeeding = dense_refeeding

        # Convolutions/Encoder
        cnn_blocks = []
        hs = []
        h = math.ceil(math.sqrt(io_dim))
        for i in range(0, len(conv_filter_sizes)):
            input_channel_size = conv_filter_sizes[i - 1] if i else 1
            output_channel_size = conv_filter_sizes[i] 
            # Convolution layer
            hs.append(math.floor(h))
            convolution = torch.nn.Conv2d(input_channel_size, output_channel_size, 
                                          conv_kernel_size, padding = 1)
            h = ((h + 2 * convolution.padding[0] - convolution.dilation[0]
                  * (convolution.kernel_size[0] - 1) - 1)/ convolution.stride[0])+1
            pooling = torch.nn.MaxPool2d((conv_pooling_size, 1))
            h = ((h + 2 * pooling.padding - pooling.dilation
                  * (pooling.kernel_size[0] - 1) - 1)/ pooling.stride[0])+1
            cnn_blocks.append(convolution)
            cnn_blocks.append(pooling)
        self.encoder = torch.nn.Sequential(*cnn_blocks)

        # Deconvolutions/Decoder
        decnn_blocks = []
        for i in range(0, len(conv_filter_sizes))[::-1]:
            input_channel_size = conv_filter_sizes[i]
            output_channel_size = conv_filter_sizes[i - 1] if i else 1
            # Deconvolution  layer
            deconvolution = torch.nn.ConvTranspose2d(input_channel_size, output_channel_size, 
                                                     conv_kernel_size, padding = 1)
            up_sampling = torch.nn.Upsample((hs[i], math.ceil(math.sqrt(io_dim))))
            decnn_blocks.append(deconvolution)
            decnn_blocks.append(up_sampling)
        decnn_blocks.append(torch.nn.Flatten())

        self.decoder = torch.nn.Sequential(*decnn_blocks)

    def encode(self, x):
        h = math.ceil(math.sqrt(x.shape[-1]))
        """
        pad = torch.zeros(*x.shape[:-1], ) 
        x = torch.cat([x, pad], dim = 2)
        """
        x = F.pad(input = x, pad = (0, (h ** 2) - x.shape[-1], 0, 0), 
                  mode = "constant", value = 0)
        x = x.view(x.shape[0], 1, h, h)

        return self.encoder(x)

    def forward(self, x):
        actual_length = x.shape[-1]
        for i in range(self.dense_refeeding):
            x = self.decoder(self.encode(x))
            x = x[:, :actual_length]

        return x




        
