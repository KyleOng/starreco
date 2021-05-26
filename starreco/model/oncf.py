import math

import torch
import torch.nn.functional as F

from .module import BaseModule
from .layers import FeaturesEmbedding, ActivationFunction, MultilayerPerceptrons
from .mixins import FeaturesEmbeddingMixin

# Done
class ONCF(BaseModule, FeaturesEmbeddingMixin):
    """
    Outer Product-based Neural Collaborative Filtering.

    - field_dims (list): List of features dimensions.
    - embed_dim (int): Embedding dimension.
    - filter_size (int): Convolution filter/depth/channel size. Default: 32.
    - kernel_size (int): Convolution square-window size or convolving square-kernel size. Default: 2
    - stride (int): Convolution stride. Default: 2.
    - activation (str): Activation function applied across the convolution layers. Default: relu.
    - batch_norm (bool): If True, apply batch normalization after c. Batch normalization is applied between activation and dropout layer across the convolution layers. Default: True.
    - lr (float): Learning rate. Default: 1e-3.
    - l2_lambda (float): L2 regularization rate. Default: 1e-3.
    - criterion (F): Criterion or objective or loss function. Default: F.mse_loss.  
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
                 criterion:F = F.mse_loss):
        super().__init__(lr, weight_decay, criterion)
        self.save_hyperparameters()
    

        # Embedding layer
        self.features_embedding = FeaturesEmbedding(field_dims, embed_dim)

        # Convolution neural network layers
        cnn_blocks = [torch.nn.LayerNorm(embed_dim)]
        # The number of convolutions = math.ceil(math.log(embed_dim, kernel_size))
        for i in range(math.ceil(math.log(embed_dim, kernel_size))):
            # 1st convolution layer does not have any depth
            input_channel_size = filter_size if i else 1 
            output_channel_size = filter_size
            # Convolution 
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
        # Flatten
        cnn_blocks.append(torch.nn.Flatten())
        # Fully connected layer
        # The author specified that there are only 2 layers (input and output layers) in the FC layer, 
        # as 1 layer MLP in NCF has more parameters than several layers of convolution in ONCF, 
        # which makes  it more stable and generalizable than MLP in NCF.
        fully_connected_net = MultilayerPerceptrons(input_dim = filter_size,
                                                    output_layer = "relu")
        cnn_blocks.append(fully_connected_net)
        self.cnn = torch.nn.Sequential(*cnn_blocks)

    def forward(self, x):
        # Generate embeddings
        user_embed, item_embed = self.user_item_embeddings(x)

        # Outer product between user and items embeddings
        outer = torch.bmm(user_embed.unsqueeze(2), item_embed.unsqueeze(1))

        # Unsqueeze outer product so that each matrix contain single depth for convolution
        outer = torch.unsqueeze(outer, 1)
        
        # Prediction
        y = self.cnn(outer)

        return y