import math

import torch
import torch.nn.functional as F

from .mf import MF
from .layers import ActivationFunction, MultilayerPerceptrons

# Done
class ONCF(MF):
    """
    Outer Product-based Neural Collaborative Filtering.

    - field_dims (list): List of features dimensions.
    - embed_dim (int): Embedding dimension.
    - filter_size (int): Convolution filter/depth/channel size. Default: 32.
    - kernel_size (int): Convolution square-window size or convolving square-kernel size. Default: 2
    - stride (int): Convolution stride. Default: 2.
    - activation (str): Activation function applied across the convolution layers. Default: "relu".
    - batch_norm (bool): If True, apply batch normalization during convolutions. Batch normalization is applied between activation and dropout layer across the convolution layers. Default: True.
    - lr (float): Learning rate. Default: 1e-3.
    - weight_decay (float): L2 regularization rate. Default: 1e-3.
    - criterion: Criterion or objective or loss function. Default: F.mse_loss.  
    """

    def __init__(self, field_dims:list, 
                 embed_dim:int = 32, #or 64
                 filter_size:int = 32, #or 64
                 kernel_size:int = 2, 
                 stride:int = 2,
                 activation:str = "relu", 
                 dropout:bool = 0.5,
                 batch_norm:bool = True, 
                 lr:float = 1e-3,
                 weight_decay:float = 1e-3,
                 criterion = F.mse_loss):
        super().__init__(field_dims, embed_dim, lr, weight_decay, criterion)
        self.save_hyperparameters()

        # Convolution neural network layers
        cnn_blocks = [torch.nn.LayerNorm(embed_dim)]
        # The number of convolutions = math.ceil(math.log(embed_dim, kernel_size))
        input_channel_size = 1
        for i in range(math.ceil(math.log(embed_dim, kernel_size))):
            # Convolution 
            output_channel_size = filter_size
            cnn_blocks.append(torch.nn.Conv2d(input_channel_size, 
                                              output_channel_size, 
                                              kernel_size, 
                                              stride = stride))
            # Activation function
            cnn_blocks.append(ActivationFunction(activation))
            # Batch normalization
            if batch_norm:
                cnn_blocks.append(torch.nn.BatchNorm2d(output_channel_size))
            if dropout > 0 and dropout <= 1:
                    cnn_blocks.append(torch.nn.Dropout(dropout))

            input_channel_size = output_channel_size
            
        # Flatten
        cnn_blocks.append(torch.nn.Flatten())
        # Fully connected layer
        """
        IMPORTANT: The author specified that there are only 2 layers (input and output layers) in the FC layer.
        1 layer MLP in NCF has more parameters than several layers of convolution in ONCF,  which makes it more stable and generalizable than MLP in NCF.
        """
        fully_connected_net = MultilayerPerceptrons(input_dim = filter_size,
                                                    output_layer = "relu")
        cnn_blocks.append(fully_connected_net)
        self.cnn = torch.nn.Sequential(*cnn_blocks)

    def forward(self, x):
        # Outer product between user and items embeddings
        x_embed = self.features_embedding(x.int())
        user_embed, item_embed = x_embed[:, 0], x_embed[:, 1]
        outer = torch.bmm(user_embed.unsqueeze(2), item_embed.unsqueeze(1))
        # Unsqueeze outer product so that each matrix contain single depth for convolution
        outer = torch.unsqueeze(outer, 1)
        
        # Non linear on outer product
        y = self.cnn(outer)

        return y