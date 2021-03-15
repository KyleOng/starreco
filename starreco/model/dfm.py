import torch
import torch.nn.functional as F

from starreco.model import FeaturesEmbedding, FeaturesLinear, PairwiseInteraction, MultilayerPerceptrons, Module

class DFM(Module):
    def __init__(self, features_dim, embed_dim, 
                 output_layers, activations, dropouts, 
                 batch_normalization = True, criterion = F.mse_loss):
        super().__init__()
        self.criterion = criterion

        # Embedding layer
        self.embedding = FeaturesEmbedding(features_dim, embed_dim)

        # Linear layer 
        self.linear = FeaturesLinear(features_dim)

        # Pairwise interaction
        self.pwin = PairwiseInteraction()

        # Multilayer perceptrons
        output_layers.insert(0, len(features_dim) * embed_dim)
        self.mlp = MultilayerPerceptrons(output_layers, activations, dropouts,
                                         batch_normalization)
        
    def forward(self, x):
        # Generate embeddings
        embed_x = self.embedding(x)

        # Prediction
        return self.linear(x) + self.pwin(embed_x) + \
        self.mlp(torch.flatten(embed_x, start_dim = 1)) 