import torch
import torch.nn.functional as F

from starreco.model import FeaturesEmbedding, Module

class MF(Module):
    def __init__(self, features_dim, embed_dim, criterion = F.mse_loss):
        super().__init__()
        self.criterion = criterion

        # Embedding layer
        self.embedding = FeaturesEmbedding(features_dim, embed_dim)

    def forward(self, x):
        # Generate embeddings
        x = self.embedding(x)
        user_embedding = x[:, 0]
        item_embedding = x[:, 1]

        # Dot product between embeddings
        matrix = torch.mm(user_embedding, item_embedding.T)

        # Obtain diagonal part of the outer product result
        diagonal = matrix.diag()

        # Reshape to match evaluation shape
        return diagonal.view(diagonal.shape[0], -1)
