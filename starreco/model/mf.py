import torch
import torch.nn.functional as F

from starreco.model import (FeaturesEmbedding, 
                            Module)

class MF(Module):
    """
    Matrix Factorization
    """
    def __init__(self, features_dim:list, 
                 embed_dim:int = 8, 
                 lr:float = 1e-3,
                 weight_decay:float = 1e-3,
                 criterion:F = F.mse_loss):
        """
        Hyperparameters setting.

        :param features_dim (list): List of feature dimensions. 

        :param embed_dim (int): Embeddings dimensions. Default: 8

        :param lr (float): Learning rate. Default: 1e-3

        :param weight_decay (float): L2 regularization weight decap: Default: 1e-3

        :param criterion (F): Objective function. Default: F.mse_loss
        """
        super().__init__(lr, weight_decay, criterion)

        # Embedding layer
        self.embedding = FeaturesEmbedding(features_dim, embed_dim)

    def forward(self, x):
        """
        Perform operations.

        :x (torch.tensor): Input tensors of shape (batch_size, 2) user and item.

        :return (torch.tensor): Output prediction tensors of shape (batch_size, 1)
        """
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
