import math

import torch
import torch.nn.functional as F

from starreco.model import FeaturesEmbedding, ActivationFunction, MultilayerPerceptrons, Module

class ConvolutionalNeuralCollaborativeFiltering(Module):
    def __init__(self, features_dim, embed_dim, 
                 channel_size, kernel_size, strides,
                 convolution_activation, fc_activation,
                 batch_normalization = True, criterion = F.mse_loss):
        super().__init__()
        self.criterion = criterion

        # Embedding layer
        self.embedding = FeaturesEmbedding(features_dim, embed_dim)

        # Convolution layers
        convolution_blocks = []
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

        # Fully connected layer (strictly one layer)
        self.fc = MultilayerPerceptrons([channel_size,1], [fc_activation], [])
        
    def forward(self, x):
        # Generate embeddings
        x = self.embedding(x)
        user_embedding = x[:, 0]
        item_embedding = x[:, 1]

        # Outer products on embeddings
        outer_product = torch.bmm(user_embedding.unsqueeze(2), item_embedding.unsqueeze(1))

        # Convolution on outer products
        outer_product = torch.unsqueeze(outer_product, 1)
        feature_map = self.convolution(outer_product)

        # Prediction
        return self.fc(feature_map)