import math

import torch
import torch.nn.functional as F

from starreco.model import FeaturesEmbedding, MultilayerPerceptrons, Module

class ConvolutionalNeuralCollaborativeFiltering(Module):
    def __init__(self, features_dim, embed_dim, 
                 channel_size, kernel_size, strides,
                 cnn_activation, fc_activation,
                 batch_normalization = True, criterion = F.mse_loss):
        super().__init__()
        self.criterion = criterion
        self.embedding = FeaturesEmbedding(features_dim, embed_dim)
        # Convolutions
        cnn_blocks = []
        for i in range(math.ceil(math.log(embed_dim, kernel_size))):
            input_channel_size = channel_size if i else 1
            output_channel_size = channel_size
            cnn_blocks.append(torch.nn.Conv2d(input_channel_size, output_channel_size, 
                                        kernel_size, stride = 2))
            cnn_blocks.append(MultilayerPerceptrons.activation(None, cnn_activation))
            if batch_normalization:
              cnn_blocks.append(torch.nn.BatchNorm2d(output_channel_size))
        cnn_blocks.append(torch.nn.Flatten())
        self.cnn = torch.nn.Sequential(*cnn_blocks)
        # Fully connected (strictly one layer)
        self.fc = MultilayerPerceptrons([channel_size,1], [fc_activation], [])
        
    def forward(self, x):
        x = self.embedding(x)
        user_embedding = x[:, 0]
        item_embedding = x[:, 1]
        # Outer product (remove .diag from torch.matmul)
        outer_product = torch.bmm(user_embedding.unsqueeze(2), item_embedding.unsqueeze(1))
        # Convolution: obtain feature maps
        outer_product = torch.unsqueeze(outer_product, 1)
        feature_map = self.cnn(outer_product)
        # Fully connected layers: prediction
        return self.fc(feature_map)