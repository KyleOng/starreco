import torch
import torch.nn.functional as F

from starreco.model import FeaturesEmbedding, Module

class MatrixFactorization(Module):
    def __init__(self, features_dim, embed_dim, criterion = F.mse_loss):
        super().__init__()
        self.criterion = criterion
        self.embedding = FeaturesEmbedding(features_dim, embed_dim)

    def forward(self, x):
        x = self.embedding(x)
        user_embedding = x[:, 0]
        item_embedding = x[:, 1]
        # Full matrix
        matrix = torch.mm(user_embedding, item_embedding.T)
        # Obtain diagonal part of matrix as result
        diagonal = matrix.diag()
        # Reshape to match evaluation shape
        return diagonal.view(diagonal.shape[0], -1)
