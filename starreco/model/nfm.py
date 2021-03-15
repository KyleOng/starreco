import torch
import torch.nn.functional as F

from starreco.model import FeaturesEmbedding, FeaturesLinear, PairwiseInteraction, MultilayerPerceptrons, Module

class NFM(Module):
    def __init__(self, features_dim, embed_dim, 
                 output_layers, activations, dropouts, 
                 fm_dropout = 0, batch_normalization = True, criterion = F.mse_loss):
        super().__init__()
        self.criterion = criterion

        # Embedding layer
        self.embedding = FeaturesEmbedding(features_dim, embed_dim)

        # Linear layer
        self.linear = FeaturesLinear(features_dim)

        # Bi-interaction layer
        self.bi = torch.nn.Sequential(
            PairwiseInteraction(reduce_sum = False),
            torch.nn.BatchNorm1d(embed_dim),
            torch.nn.Dropout(fm_dropout)
        )
        output_layers.insert(0, embed_dim)

        # Multilayer Perceptrons
        self.mlp = MultilayerPerceptrons(output_layers, activations, dropouts,
                                         batch_normalization)
        
    def forward(self, x):
        # Generate embeddings
        embed_x = self.embedding(x)

        # Bi-interaction
        cross_term = self.bi(embed_x)

        # Prediction
        return self.linear(x) + self.mlp(cross_term)