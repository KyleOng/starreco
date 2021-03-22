import math

import numpy as np
import torch
import torch.nn.functional as F

from starreco.model import (FeaturesEmbedding, 
                            FeaturesLinear, 
                            ActivationFunction,
                            MultilayerPerceptrons, 
                            CompressedInteraction, 
                            InnerProduct, 
                            Module)

class FGCNN(Module):
    """
    Feature Generation Convolutional Neurak Network
    """
    def __init__(self, feature_dims:list, multivalent_fields:list, 
                 multivalent_function:str = "sum", 
                 embed_dim:int = 40, 
                 conv_filter_sizes:list = [6, 8, 10, 12], 
                 conv_kernel_size:int = 100, 
                 conv_pooling_size:int = 2, 
                 conv_activations:str = "tanh", 
                 recbm_filter_size:int = 3,  
                 recbm_activations:str = "tanh",
                 nn_hidden_dims:list = [2048, 1024, 512, 256, 128], 
                 nn_activations:list = "relu", 
                 nn_dropouts:list = 0.5, 
                 batch_norm:bool = True, 
                 lr:float = 1e-3,
                 weight_decay:float = 1e-3,
                 criterion:F = F.mse_loss):
        """
        Hyperparameters setting.      
        
        :param feature_dims (list): List of feature dimensions.

        :param multivalent_fields (list): List of multivalent fields range, where 1 = univalent and >1 = multivalent, maps to feature_dims and later combine using the multivalent_function. The sum(multivalent_field) = len(feature_sdim) 

        :param multivalent_function (str): Function which to combine multivalent fields. Option: ["sum", "avg"]. Default: "sum"

        :param embed_dim (int): Embedding size. Default: 32 or 64

        :param conv_filter_size (int): Convolution filter/depth/channel size. Default: 32 or 64

        :param conv_kernel_size (int): Convolution kernel/window size. Default: 2

        :param conv_pooling_size (int): Convolution max pooling size. Default: 2

        :param conv_activation (str): Convolution activation function. Default: "relu"

        :recbm_filter_size (int): New filter/depth/channel size for recombination layer. Default: 3

        :recbm_filter_size (int): New filter/depth/channel size for recombination layer. Default: 3

        :param nn_hidden_dims (list): List of number of hidden nodes for inner product neural network. Default: [2048, 1024, 512, 256, 128, 1]

        :param nn_activations (str/list): Activation function for inner product neural network. Default: "relu"

        :param fc_dropouts (float/list): Dropout for inner product neural network. Default: 0.5

        :param batch_norm (bool): If True, apply batch normalization on every hidden layer. Default: True

        :param lr (float): Learning rate. Default: 1e-3

        :param weight_decay (float): L2 regularization weight decay: Default: 1e-3

        :param criterion (F): Objective function. Default: F.mse_loss
        """
        super().__init__(lr, weight_decay, criterion)
        self.multivalent_fields = multivalent_fields

        # Embedding layer
        self.embedding = FeaturesEmbedding(feature_dims, embed_dim)

        # Choose multivalent function operation function
        if multivalent_function == "sum":
            self.multivalent_function = torch.sum
        elif multivalent_function == "avg":
            self.multivalent_function = torch.mean
        else:
            raise Exception("Function incorrect. Choose between 'sum' or 'avg'")

        # Convolutions, pooling and recombination layers for features generation
        self.convolutions = torch.nn.ModuleList()
        self.poolings = torch.nn.ModuleList()
        self.recombinations = torch.nn.ModuleList()
        new_features_size = 0
        h = v_pad = len(multivalent_fields)
        w = embed_dim
        for i in range(0, len(conv_filter_sizes)):
            input_channel_size = conv_filter_sizes[i - 1] if i else 1
            output_channel_size = conv_filter_sizes[i] 
            # Convolution layer
            v_pad = int((v_pad - (v_pad - conv_kernel_size + 1))/ 2)
            convolution = torch.nn.Conv2d(input_channel_size, output_channel_size, 
                                          (conv_kernel_size, 1), padding = (v_pad, 0))
            sequential_blocks = [convolution]
            if batch_norm:
                sequential_blocks.append(torch.nn.BatchNorm2d(output_channel_size))
            sequential_blocks.append(ActivationFunction(conv_activations))
            self.convolutions.append(torch.nn.Sequential(*sequential_blocks))
            h = ((h + 2 * convolution.padding[0] - convolution.dilation[0]
                  * (convolution.kernel_size[0] - 1) - 1)/ convolution.stride[0])+1
            w = ((w + 2 * convolution.padding[1] - convolution.dilation[1]
                  * (convolution.kernel_size[1] - 1) - 1)/ convolution.stride[1])+1
            # Pooling layer
            pooling = torch.nn.MaxPool2d((conv_pooling_size, 1))
            h = ((h + 2 * pooling.padding - pooling.dilation
                  * (pooling.kernel_size[0] - 1) - 1)/ pooling.stride[0])+1
            w = ((w + 2 * pooling.padding - pooling.dilation
                  * (pooling.kernel_size[1] - 1) - 1)/ pooling.stride[1])+1
            self.poolings.append(pooling)
            # Recombination layer
            self.recombinations.append(
                torch.nn.Sequential(
                    torch.nn.Flatten(),
                    torch.nn.Linear(output_channel_size * math.floor(h) * math.floor(w),
                                    recbm_filter_size * math.floor(h) * math.floor(w)),
                    ActivationFunction(recbm_activations)
                )
            )
        # Calculate inner product neural network input size
            new_features_size += (recbm_filter_size * math.floor(h) * math.floor(w))
        new_embed_x_size = embed_dim * len(multivalent_fields)
        combine_input_size = new_features_size + new_embed_x_size
        ip_result_size = ((combine_input_size/embed_dim)*((combine_input_size/embed_dim)-1) 
        / 2) * embed_dim
        nn_input_size = int(combine_input_size + ip_result_size)

        # Inner product (set reduce_sum as False)
        self.ip = InnerProduct(False)

        # Multilayer perceptrons
        self.nn = MultilayerPerceptrons(input_dim = nn_input_size,
                                        hidden_dims = nn_hidden_dims, 
                                        activations = nn_activations, 
                                        dropouts = nn_dropouts,
                                        output_layer = "relu",
                                        batch_norm = batch_norm)

    def forward(self, x):
        """
        Perform operations.

        :x (torch.tensor): Input tensors of shape (batch_size, len(feature_dims)).

        :return (torch.tensor): Output prediction tensors of shape (batch_size, 1)
        """
        # Generate embeddings
        embed_x = self.embedding(x)

        # Multivalent features operations (sum or avg)
        for i, field_range in enumerate(self.multivalent_fields):
            start_i = np.cumsum(self.multivalent_fields)[i - 1] if i else 0
            end_i = np.cumsum(self.multivalent_fields)[i]
            embed_x_sub = embed_x[:, start_i:end_i]
            if field_range > 1:
                embed_x_sub = self.multivalent_function(embed_x_sub, 1, keepdim = True)
            if i == 0:
                new_embed_x = embed_x_sub
            else:
                new_embed_x = torch.cat((new_embed_x, embed_x_sub), 1)
        
        # Feature generations using CNN
        fg_x = new_embed_x.unsqueeze(1)
        new_features_list = []
        for i in range(len(self.convolutions)):
            fg_x = self.convolutions[i](fg_x)
            fg_x = self.poolings[i](fg_x)
            new_features = self.recombinations[i](fg_x)
            new_features = new_features.view(-1, 
                                             int(new_features.shape[1]/new_embed_x.shape[-1]), 
                                             new_embed_x.shape[-1])
            new_features_list.append(new_features)
        new_features = torch.cat(new_features_list, dim = 1)

        # Deep factorization machine
        combine_input = torch.cat([new_embed_x, new_features], dim = 1)

        # Pairwise interaction inner product 
        ip_result = self.ip(torch.unbind(combine_input.unsqueeze(2), dim = 1))

        # Flatten and concatenate
        combine_input = torch.flatten(combine_input, start_dim = 1)
        ip_result = torch.flatten(ip_result, start_dim = 1)
        nn_input = torch.cat([combine_input, ip_result], dim = 1)
        
        # Prediction
        return self.nn(nn_input)