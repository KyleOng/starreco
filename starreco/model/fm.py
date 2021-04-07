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
    def __init__(self, field_dims:list, 
                 embed_dim:int = 8, 
                 lr:float = 1e-3,
                 weight_decay:float = 1e-3,
                 criterion:F = F.mse_loss,
                 save_hyperparameters:bool = True):
        """
        Hyperparameters setting.

        :param field_dims (list): List of field dimensions. 

        :param embed_dim (int): Embeddings dimensions. Default: 8

        :param lr (float): Learning rate. Default: 1e-3

        :param weight_decay (float): L2 regularization weight decay: Default: 1e-3

        :param criterion (F): Objective function. Default: F.mse_loss
        """
        super().__init__(lr, weight_decay, criterion)

        # Embedding layer
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)

        # Linear layer
        self.linear = FeaturesLinear(field_dims)

        # Pairwise interaction
        self.pairwise_interaction = PairwiseInteraction()

    def forward(self, x):
        """
        Perform operations.

        :x (torch.tensor): Input tensors of shape (batch_size, len(field_dims))

        :return (torch.tensor): Output prediction tensors of shape (batch_size, 1)
        """
        # Generate embeddings
        embed_x = self.embedding(x)

        # Prediction
        return self.linear(x) + self.pairwise_interaction(embed_x)