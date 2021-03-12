import torch
import torch.nn.functional as F

from starreco.model import FeaturesEmbedding, FeaturesLinear, PairwiseInteraction, Module

class FactorizationMachine(Module):
    def __init__(self, features_dim, embed_dim, criterion = F.mse_loss):
        super().__init__()
        self.criterion = criterion

        # Embedding layer
        self.embedding = FeaturesEmbedding(features_dim, embed_dim)

        # Linear layer
        self.linear = FeaturesLinear(features_dim)

        # Pairwise interaction
        self.pairwise_interaction = PairwiseInteraction()

    def forward(self, x):
        # Generate embeddings
        embed_x = self.embedding(x)

        # Prediction
        return self.linear(x) + self.pairwise_interaction(embed_x)