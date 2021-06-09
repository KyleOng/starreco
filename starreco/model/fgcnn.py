import math
import time

import numpy as np
import torch
import torch.nn.functional as F

from .module import BaseModule
from .layers import FeaturesEmbedding, MultilayerPerceptrons, ActivationFunction, InnerProduct


class FGCNN(BaseModule):
    """
    Feature Generation Convolutional Neural Network.
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
                 net_activations:list = "relu", 
                 net_dropouts:list = 0.5, 
                 lr:float = 1e-3,
                 weight_decay:float = 1e-6,
                 criterion = F.mse_loss):
        
        super().__init__(lr, weight_decay, criterion)
        self.save_hyperparameters()

        self.user_features_split = user_features_split
        self.item_features_split = item_features_split

        self.features_embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.user_features_embedding = FeaturesEmbedding(user_features_split, embed_dim)
        self.item_features_embedding = FeaturesEmbedding(item_features_split, embed_dim)

        self.cnn = torch.nn.ModuleList()
        self.recombinations = torch.nn.ModuleList()
        new_features_size = 0

        height = v_pad = len(user_features_split) + len(item_features_split)
        width = embed_dim
        
        input_channel_size = 1
        for i in range(len(cnn_filter_sizes)):
            v_pad = int((v_pad - (v_pad - cnn_kernel_size + 1))/ 2)
            
            # convolution layer
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

            #print(height, width)

            # Pooling layer
            pooling = torch.nn.MaxPool2d((cnn_pooling_size, 1))
            self.cnn.append(pooling)
            # Update height and width from pooling layer
            height  = ((height + 2 * pooling.padding - pooling.dilation * (pooling.kernel_size[0] - 1) - 1)/ pooling.stride[0]) + 1
            width = ((width + 2 * pooling.padding - pooling.dilation  * (pooling.kernel_size[1] - 1) - 1)/ pooling.stride[1]) + 1

            #print(height, width)

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
            new_features_size += recombination_output_size

        # Input size for inner product-based neural network
        new_embed_size = embed_dim * (len(user_features_split) + len(item_features_split))
        combine_input_size = new_features_size + new_embed_size
        inner_product_output_size = (combine_input_size / embed_dim ) * ((combine_input_size / embed_dim) - 1) * embed_dim
        net_input_size = int(combine_input_size + inner_product_output_size)

        # Inner product-based neural network
        inner_product = InnerProduct(False)
        net = MultilayerPerceptrons(input_dim = net_input_size,
                                    hidden_dims = net_hidden_dims, 
                                    activations = net_activations, 
                                    dropouts = net_dropouts,
                                    output_layer = "relu")
        self.inner_product_net = torch.nn.Sequential(inner_product, net)

    def forward(self, x, user_x, item_x):

        batch_size = x.shape[0]

        user_start = 0
        user_features_embed = torch.zeros(0).to(self.device)
        for i, user_feature_split in enumerate(self.user_features_split):
            pad = (i, len(self.user_features_split)-i-1)
            user_end = user_feature_split
            user_end += user_start
            user_rows, user_feature_indices = (user_x[:, user_start:user_end] == 1).nonzero(as_tuple = True)
            user_feature_indices_pad = F.pad(input = user_feature_indices.unsqueeze(1), pad = pad, mode = "constant", value = 0)
            user_feature_embed = self.user_features_embedding(user_feature_indices_pad.int())[:, i]
            if batch_size != len(user_rows):
                current_user_row = user_rows[0]
                user_row_start = 0
                user_rows = torch.cat((user_rows, torch.tensor([-1])))

                new_user_feature_embed = torch.zeros(0).to(self.device)
                for i in range(1, len(user_rows)):
                    next_user_row = user_rows[i]
                    if current_user_row != next_user_row:
                        user_row_end = i
                        user_feature_embed_sum = torch.sum(user_feature_embed[user_row_start: user_row_end, :], dim = 0)
                        new_user_feature_embed = torch.cat((new_user_feature_embed, user_feature_embed_sum.unsqueeze(0)))
                        user_row_start = i
                    current_user_row = next_user_row
                user_feature_embed = new_user_feature_embed
            user_features_embed = torch.cat((user_features_embed, user_feature_embed.unsqueeze(1)), dim = 1)
            user_start += user_feature_split

        item_start = 0
        item_features_embed = torch.zeros(0).to(self.device)
        for i, item_feature_split in enumerate(self.item_features_split):
            pad = (i, len(self.item_features_split)-i-1)
            item_end = item_feature_split
            item_end += item_start
            item_rows, item_feature_indices = (item_x[:, item_start:item_end] == 1).nonzero(as_tuple = True)
            item_feature_indices_pad = F.pad(input = item_feature_indices.unsqueeze(1), pad = pad, mode = "constant", value = 0)
            item_feature_embed = self.item_features_embedding(item_feature_indices_pad.int())[:, i]
            if batch_size != len(item_rows):
                current_item_row = item_rows[0]
                item_row_start = 0
                item_rows = torch.cat((item_rows, torch.tensor([-1]).to(self.device)))

                new_item_feature_embed = torch.zeros(0).to(self.device)
                for i in range(1, len(item_rows)):
                    next_item_row = item_rows[i]
                    if current_item_row != next_item_row:
                        item_row_end = i
                        item_feature_embed_sum = torch.sum(item_feature_embed[item_row_start: item_row_end, :], dim = 0)
                        new_item_feature_embed = torch.cat((new_item_feature_embed, item_feature_embed_sum.unsqueeze(0)))
                        item_row_start = i
                    current_item_row = next_item_row
                item_feature_embed = new_item_feature_embed
            item_features_embed = torch.cat((item_features_embed, item_feature_embed.unsqueeze(1)), dim = 1)
            item_start += item_feature_split

        x_embed = self.features_embedding(x.int())
        user_embed, item_embed = x_embed[:, 0], x_embed[:, 1]
        dot = torch.sum(user_embed * item_embed, dim = 1)
        
        # Reshape to match target shape
        y = dot.view(dot.shape[0], -1)

        return y

        #(torch.tensor([0,1,0,1,0]) == 1.).nonzero().flatten().shape[0] == 2




