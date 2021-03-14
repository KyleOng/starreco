import math

import torch
import torch.nn.functional as F

from starreco.model import FeaturesEmbedding, ActivationFunction, MultilayerPerceptrons, Module

class ConvolutionalDeepCollaborativeFiltering(Module):
    def __init__(self, features_dim, embed_dim, 
                 channel_size, kernel_size, strides,
                 convolution_activation, fc_activation,
                 batch_normalization = True, criterion = F.mse_loss):
        super().__init__()
        self.criterion = criterion

        # Embedding layer
        self.embedding = FeaturesEmbedding(features_dim, embed_dim)

        # Fully connected layer before convolutions
        self.fc_before = MultilayerPerceptrons([embed_dim ** 2, 
                                          embed_dim ** 2], 
                                         [fc_activation], 
                                         [])

        # Convolution layers
        convolution_blocks = [torch.nn.LayerNorm(embed_dim)]
        for i in range(math.ceil(math.log(embed_dim, kernel_size))):
            input_channel_size = channel_size if i else 1
            output_channel_size = channel_size
            # Convolution 
            convolution_blocks.append(torch.nn.Conv2d(input_channel_size, output_channel_size, 
            kernel_size, stride = 2))
            # Activation function
            convolution_blocks.append(ActivationFunction(convolution_activation))
            # Batch normalization
            if batch_normalization:
              convolution_blocks.append(torch.nn.BatchNorm2d(output_channel_size))
        # Flatten
        convolution_blocks.append(torch.nn.Flatten())
        self.convolution = torch.nn.Sequential(*convolution_blocks)

        # Fully connected layer after convolutions
        self.fc_after = MultilayerPerceptrons([channel_size,1], [fc_activation], [])
        
    def forward(self, x):
        # Generate embeddings
        x = self.embedding(x)
        user_embedding = x[:, 0]
        item_embedding = x[:, 1]

        # Outer products on embeddings
        residual = torch.bmm(user_embedding.unsqueeze(2), item_embedding.unsqueeze(1))

        # FC before convolution
        flat = torch.flatten(residual, start_dim = 1)
        out = self.fc_before(flat).view(residual.shape)

        # Residual connection
        out += residual

        # Convolution on outer products
        out = torch.unsqueeze(residual, 1)
        feature_map = self.convolution(out)

        # FC after convolution
        return self.fc_after(feature_map)