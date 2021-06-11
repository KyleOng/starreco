import math
from typing import Union

import torch
import torch.nn.functional as F

from .module import BaseModule
from .layers import FeaturesEmbedding, MultilayerPerceptrons, ActivationFunction, InnerProduct


class FGCNN(BaseModule):
    """
    Feature Generation Convolutional Neural Network with Inner Product Neural Network.

    - user_features_split (list): List of integers which determine how user features are splitted. \
    - item_features_split (list): List of integers which determine how item features are splitted. 
    - field_dims (list): List of field dimensions.
    - embed_dim (int): . Default: 40.
    - rec_filter_size (int): . Default: 3.
    - cnn_filter_size (list): . Default: [6].
    - cnn_kernel_size (int): . Default: 100.
    - cnn_pooling_size (int): . Default: 2.
    - cnn_activation (str): . Default: "tanh".
    - net_hidden_dims (list): . Default: [256, 128].
    - net_activation (str): . Default: "relu".
    - net_dropout (int/float): . Default: 0.5.
    - lr (float): Learning rate. Default: 1e-3.
    - weight_decay (float): L2 regularization rate. Default: 1e-3.
    - criterion: Criterion or objective or loss function. Default: F.mse_loss.
    """

    def __init__(self,
                 user_features_split:list,
                 item_features_split:list,
                 field_dims:list,
                 embed_dim:int = 40,
                 rec_filter_size:int = 3,
                 cnn_filter_sizes:list = [6],
                 cnn_kernel_size:int = 100,
                 cnn_pooling_size:int = 2,
                 cnn_activation:str = "tanh",
                 net_hidden_dims:list = [256, 128], 
                 net_activation:str = "relu", 
                 net_dropout:list = 0.5, 
                 lr:float = 1e-3,
                 weight_decay:float = 1e-6,
                 criterion = F.mse_loss):
        
        super().__init__(lr, weight_decay, criterion)
        self.save_hyperparameters()

        self.user_features_split = user_features_split
        self.item_features_split = item_features_split

        # Field embedding layer
        self.features_embedding = FeaturesEmbedding(field_dims, embed_dim)

        # User feature embedding layer
        self.user_features_embedding = torch.nn.ModuleList()
        for user_feature_split in user_features_split:
            self.user_features_embedding.append(torch.nn.Linear(user_feature_split, embed_dim, bias = False))

        # Item feature embedding layer
        self.item_features_embedding = torch.nn.ModuleList()
        for item_feature_split in item_features_split:
            self.item_features_embedding.append(torch.nn.Linear(item_feature_split, embed_dim, bias = False))

        # Convolution neural network layers
        self.cnn = torch.nn.ModuleList()
        self.recombinations = torch.nn.ModuleList()
        height = v_pad = len(user_features_split) + len(item_features_split) + 2
        width = embed_dim
        embed_size = height * width
        input_channel_size = 1
        for i in range(len(cnn_filter_sizes)):
            v_pad = int((v_pad - (v_pad - cnn_kernel_size + 1))/ 2)
            
            # Convolution layer
            output_channel_size = cnn_filter_sizes[i]
            convolution = torch.nn.Conv2d(input_channel_size,
                                          output_channel_size,
                                          (cnn_kernel_size , 1),
                                          padding = (v_pad, 0))
            self.cnn.append(convolution)
            # Update height and width from convolution layer
            height = ((height + 2 * convolution.padding[0] - convolution.dilation[0] * (convolution.kernel_size[0] - 1) - 1) / convolution.stride[0]) + 1
            width = ((width + 2 * convolution.padding[1] - convolution.dilation[1] * (convolution.kernel_size[1] - 1) - 1) / convolution.stride[1]) + 1
            # convolution layer activation function
            if cnn_activation.lower() != "linear":
                activation_function = ActivationFunction(cnn_activation)
                self.cnn.append(activation_function)

            # Pooling layer
            pooling = torch.nn.MaxPool2d((cnn_pooling_size, 1))
            self.cnn.append(pooling)
            # Update height and width from pooling layer
            height  = ((height + 2 * pooling.padding - pooling.dilation * (pooling.kernel_size[0] - 1) - 1)/ pooling.stride[0]) + 1
            width = ((width + 2 * pooling.padding - pooling.dilation  * (pooling.kernel_size[1] - 1) - 1)/ pooling.stride[1]) + 1

            # Recombination layer
            recombination_input_size = output_channel_size * math.floor(height) * math.floor(width)
            recombination_output_size = rec_filter_size * math.floor(height) * math.floor(width)
            recombination_blocks = [torch.nn.Flatten(), 
                                    torch.nn.Linear(recombination_input_size, recombination_output_size)]
            # Recombination layer activation function
            if cnn_activation.lower() != "linear":
                activation_function = ActivationFunction(cnn_activation)
                recombination_blocks.append(activation_function)
            recombination = torch.nn.Sequential(*recombination_blocks)
            self.recombinations.append(recombination)

            # Calculate inner product neural network input size
            embed_size += recombination_output_size

            input_channel_size = output_channel_size

        # Input size for inner product-based neural network
        num_pairs = (embed_size / embed_dim)
        inner_product_output_size = (num_pairs * (num_pairs - 1) / 2) * embed_dim
        net_input_size = int(embed_size + inner_product_output_size)

        # Inner product-based neural network
        self.inner_product = InnerProduct(False)
        self.net = MultilayerPerceptrons(input_dim = net_input_size,
                                    hidden_dims = net_hidden_dims, 
                                    activations = net_activation, 
                                    dropouts = net_dropout,
                                    output_layer = "relu")

    def forward(self, x, user_x, item_x):
        # Field embeddings
        x_embed = self.features_embedding(x.int())

        # User feature embeddings
        user_start = 0
        for i, user_feature_split in enumerate(self.user_features_split):
            user_end = user_feature_split
            user_end += user_start
            user_x_embed = self.user_features_embedding[i](user_x[:, user_start:user_end])
            x_embed = torch.cat((x_embed, user_x_embed.unsqueeze(1)), dim = 1)
            user_start += user_feature_split

        # Item feature embedding
        item_start = 0
        for i, item_feature_split in enumerate(self.item_features_split):
            item_end = item_feature_split
            item_end += item_start
            item_x_embed = self.item_features_embedding[i](item_x[:, item_start:item_end])
            x_embed = torch.cat((x_embed, item_x_embed.unsqueeze(1)), dim = 1)
            item_start += item_feature_split

        # Convolutional neural network feature extraction and recombintation
        feature = x_embed.unsqueeze(1)
        i = 0
        for module in self.cnn:
            feature = module(feature)
            if type(module) == torch.nn.MaxPool2d:
                recombination = self.recombinations[i](feature)
                recombination = recombination.view(-1, int(recombination.shape[1]/x_embed.shape[-1]), x_embed.shape[-1])
                x_embed = torch.cat((x_embed, recombination), dim = 1)
                i += 1

        # Inner product neural network
        # Inner product on recombined features
        inner_product = self.inner_product(torch.unbind(x_embed.unsqueeze(2), dim = 1))
        inner_product = torch.flatten(inner_product, start_dim = 1)
        x_embed = torch.flatten(x_embed, start_dim = 1)
        # Concatenate embeddings with inner product output
        concat = torch.cat((x_embed, inner_product), dim = 1)
        # Non linear on concatenated features
        y = self.net(concat)
        
        return y