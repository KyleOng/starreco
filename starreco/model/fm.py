import torch
import torch.nn.functional as F

from starreco.model import (FeaturesEmbedding, 
                            FeaturesLinear, 
                            PairwiseInteraction, 
                            Module)

class FM(Module):
    """
    Factorization Machine
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

        # Linear layer
        self.linear = FeaturesLinear(features_dim)

        # Pairwise interaction
        self.pairwise_interaction = PairwiseInteraction()

    def forward(self, x):
        """
        Perform operations.

        :x (torch.tensor): Input tensors of shape (batch_size, len(features_dim))

        :return (torch.tensor): Output prediction tensors of shape (batch_size, 1)
        """
        # Generate embeddings
        embed_x = self.embedding(x)

        # Prediction
        return self.linear(x) + self.pairwise_interaction(embed_x)