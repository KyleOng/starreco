import torch
import torch.nn.functional as F

from starreco.model import FeaturesEmbedding, MultilayerPerceptrons, Module

class NeuralCollaborativeFiltering(Module):
    def __init__(self, features_dim, embed_dim,
                 output_layers, activations, dropouts,
                 batch_normalization = True, criterion = F.mse_loss):
        super().__init__()
        self.criterion = criterion
        self.embedding = FeaturesEmbedding(features_dim, embed_dim)
        output_layers.insert(0, embed_dim * 2)
        self.mlp = MultilayerPerceptrons(output_layers, activations, dropouts,
                                         batch_normalization)

    def forward(self, x):
        # Concatenate embeddings
        x = self.embedding(x)
        x = torch.flatten(x, start_dim = 1)

        return self.mlp(x)