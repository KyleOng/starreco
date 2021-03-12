import torch
import torch.nn.functional as F

from starreco.model import FeaturesEmbedding, MultilayerPerceptrons, Module

class NeuralCollaborativeFiltering(Module):
    def __init__(self, features_dim, embed_dim,
                 output_layers, activations, dropouts,
                 batch_normalization = True, criterion = F.mse_loss):
        super().__init__()
        self.criterion = criterion

        # Embedding layer
        self.embedding = FeaturesEmbedding(features_dim, embed_dim)

        # MLP layers
        output_layers.insert(0, embed_dim * 2)
        self.mlp = MultilayerPerceptrons(output_layers, activations, dropouts,
                                         batch_normalization)

    def forward(self, x):
        # Generate embeddings
        x = self.embedding(x)

        # Concatenate embeddings
        x = torch.flatten(x, start_dim = 1)

        # Prediction
        return self.mlp(x)