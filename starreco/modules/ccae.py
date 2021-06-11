import math
from typing import Union

import torch
import torch.nn.functional as F

from .module import BaseModule
from .layers import ActivationFunction
from ..evaluation import masked_mse_loss

# In progress
class CCAE(BaseModule):
    """
    Collaborative Convolution Autoencoder
    
    - input_output_dim (int): Number of neurons in the input and output layer.
    - filter_sizes (list): List of convolution filter/depth/channel size. Default: [16, 8].
    - kernel_size (int): Convolution square-window size or convolving square-kernel size. Default: 3
    - e_activations (str/list): List of activation functions in the encoder layers. Default: "relu".
    - d_activations (str/list): List of activation functions in the decoder layers. Default: "relu".
    - e_dropouts (int/float/list): List of dropout values in the encoder layers. Default: 0.
    - d_dropouts (int/float/list): List of dropout values in the decoder layers. Default: 0.
    - dropout (float): Dropout value in the latent space. Default: 0.
    - batch_norm (bool): If True, apply batch normalization in all hidden layers. Default: True.
    - lr (float): Learning rate. Default: 1e-3.
    - weight_decay (float): L2 regularization rate. Default: 1e-3.
    - criterion: Criterion or objective or loss function. Default: masked_mse_loss
    """

    def __init__(self, 
                 input_output_dim:int,
                 filter_sizes:list = [16, 8], 
                 kernel_size:int = 3, 
                 pooling_size:int = 2, 
                 e_activation:str = "relu", 
                 d_activation:str = "relu",
                 e_dropout:Union[int, float] = 0,
                 d_dropout:Union[int, float] = 0, 
                 dropout:Union[int, float] = 0, 
                 batch_norm:bool = True,
                 lr:float = 1e-3,
                 weight_decay:float = 1e-3,
                 criterion = masked_mse_loss):

        super().__init__(lr, weight_decay, criterion)
        super().save_hyperparameters()
        self.input_output_dim = input_output_dim

        # Convolution/Encoder layer
        encoder_blocks = []
        heights = []
        height = math.ceil(math.sqrt(input_output_dim))
        for i in range(0, len(filter_sizes)):
            input_channel_size = filter_sizes[i - 1] if i else 1
            output_channel_size = filter_sizes[i] 

            # Convolution layer
            heights.append(math.floor(height))
            convolution = torch.nn.Conv2d(input_channel_size, 
                                          output_channel_size, 
                                          kernel_size, 
                                          padding = 1)
            encoder_blocks.append(convolution)
            height = ((height + 2 * convolution.padding[0] - convolution.dilation[0] * (convolution.kernel_size[0] - 1) - 1)/ convolution.stride[0])+1

            # Batch normalization
            if batch_norm:
                batch_normalization = torch.nn.BatchNorm2d(output_channel_size)
                encoder_blocks.append(batch_normalization)

            # Activation function
            activation_function = ActivationFunction(e_activation)
            encoder_blocks.append(activation_function)

            # Pooling layer
            pooling = torch.nn.MaxPool2d((pooling_size, 1))
            encoder_blocks.append(pooling)
            height = ((height + 2 * pooling.padding - pooling.dilation * (pooling.kernel_size[0] - 1) - 1)/ pooling.stride[0])+1

            # Dropout
            if i == len(filter_sizes)-1:
                if dropout > 0 and dropout <= 1:
                    encoder_blocks.append(torch.nn.Dropout(dropout))
            else:
                if e_dropout > 0 and e_dropout <= 1:
                    encoder_blocks.append(torch.nn.Dropout(e_dropout))

        self.encoder = torch.nn.Sequential(*encoder_blocks)

        # Deconvolution/Decoder layers
        decoder_blocks = []
        for i in range(0, len(filter_sizes))[::-1]: # Reverse filter sizes
            input_channel_size = filter_sizes[i]
            output_channel_size = filter_sizes[i - 1] if i else 1

            # Deconvolution layer
            deconvolution = torch.nn.ConvTranspose2d(input_channel_size, output_channel_size, 
                                                     kernel_size, padding = 1)
            decoder_blocks.append(deconvolution)

            # Batch normalization
            # Batch normalization not applied at output layer.
            if i and batch_norm:
                batch_normalization = torch.nn.BatchNorm2d(output_channel_size)
                decoder_blocks.append(batch_normalization)

            # Activation function
            # ReLU activation at output layer.
            if i:
                activation_function = ActivationFunction(d_activation)
            else:
                activation_function = ActivationFunction("relu")
            decoder_blocks.append(activation_function)

            # Upsampling layer
            up_sampling = torch.nn.Upsample((heights[i], math.ceil(math.sqrt(input_output_dim))))
            decoder_blocks.append(up_sampling)

            # Dropout
            # Dropout not applied at output layer.
            if i and d_dropout > 0 and d_dropout <= 1:
                decoder_blocks.append(torch.nn.Dropout(d_dropout))
        # Flatten
        decoder_blocks.append(torch.nn.Flatten())
        self.decoder = torch.nn.Sequential(*decoder_blocks)

    def encode(self, x):
        height = math.ceil(math.sqrt(x.shape[-1]))

        x = F.pad(input = x, pad = (0, (height ** 2) - x.shape[-1], 0, 0), mode = "constant", value = 0)
        x = x.view(x.shape[0], 1, height, height)

        return self.encoder(x)

    def decode(self, x):
        x = self.decoder(x)
        x = x[:, :self.input_output_dim]
        return x

    def forward(self, x):
        x = self.decode(self.encode(x))
        return x 