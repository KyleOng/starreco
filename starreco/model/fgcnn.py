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
    def __init__(self, features_dim, embed_dim, vis, 
                 filter_sizes, kernel_size, pooling_size, new_filter_size,  
                 output_layers, activations, dropouts,
                 multivalent_function = "sum", 
                 cnn_activation = "tanh", recbm_activation = "tanh", 
                 cnn_batch_norm = True, mlp_batch_norm = True, 
                 criterion = F.mse_loss):
        super().__init__()
        self.criterion = criterion
        self.vis = vis

        # Embedding layer
        self.embedding = FeaturesEmbedding(features_dim, embed_dim)

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
        h = v_pad = len(vis)
        w = embed_dim
        filter_sizes.insert(0, 1)
        for i in range(1, len(filter_sizes)):
            input_channel_size = filter_sizes[i - 1]
            output_channel_size = filter_sizes[i] 
            # Convolution layer
            v_pad = int((v_pad - (v_pad - kernel_size + 1))/ 2)
            convolution = torch.nn.Conv2d(input_channel_size, output_channel_size, 
                                          (kernel_size, 1), padding = (v_pad, 0))
            sequential_blocks = [convolution, ActivationFunction(cnn_activation)]
            if cnn_batch_norm:
                sequential_blocks.append(torch.nn.BatchNorm2d(output_channel_size))
            self.convolutions.append(torch.nn.Sequential(*sequential_blocks))
            h = ((h + 2 * convolution.padding[0] - convolution.dilation[0]
                  * (convolution.kernel_size[0] - 1) - 1)/ convolution.stride[0])+1
            w = ((w + 2 * convolution.padding[1] - convolution.dilation[1]
                  * (convolution.kernel_size[1] - 1) - 1)/ convolution.stride[1])+1
            # Pooling layer
            pooling = torch.nn.MaxPool2d((pooling_size, 1))
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
                                    new_filter_size * math.floor(h) * math.floor(w)),
                    ActivationFunction(recbm_activation)
                )
            )
        # Calculate MLP input size
            new_features_size += (new_filter_size * math.floor(h) * math.floor(w))
        new_embed_x_size = embed_dim * len(vis)
        combine_input_size = new_features_size + new_embed_x_size
        ip_result_size = ((combine_input_size/embed_dim)*((combine_input_size/embed_dim)-1) 
        / 2) * embed_dim
        mlp_input_size = int(combine_input_size + ip_result_size)

        # Inner product (set reduce_sum as False)
        self.ip = InnerProduct(False)

        # Multilayer perceptrons
        output_layers.insert(0, mlp_input_size)
        self.mlp = MultilayerPerceptrons(output_layers, activations, dropouts,
                                         mlp_batch_norm)

    def forward(self, x):
        # Generate embeddings
        embed_x = self.embedding(x)

        # Multivalent features operations (sum or avg)
        for i, vi in enumerate(self.vis):
            start_i = np.cumsum(self.vis)[i - 1] if i else 0
            end_i = np.cumsum(self.vis)[i]
            embed_x_sub = embed_x[:, start_i:end_i]
            if vi > 1:
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
        mlp_input = torch.cat([combine_input, ip_result], dim = 1)
        
        # Prediction
        return self.mlp(mlp_input)