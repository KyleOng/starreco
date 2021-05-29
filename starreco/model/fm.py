import torch
import torch.nn.functional as F

from .module import BaseModule
from .layers import FeaturesEmbedding, FeaturesLinear, PairwiseInteraction

# Done
class FM(BaseModule):
    """
    Factorization Machine.

    - field_dims (list): List of features dimensions. 
    - embed_dim (int): Embedding dimension. Default: 8.
    - lr (float): Learning rate. Default: 1e-3.
    - l2_lambda (float): L2 regularization rate. Default: 1e-3.
    - criterion (F): Criterion or objective or loss function. Default: F.mse_loss.
    """

    def __init__(self, 
                 field_dims:list, 
                 embed_dim:int = 8, 
                 lr:float = 1e-3,
                 weight_decay:float = 1e-3,
                 criterion:F = F.mse_loss):
        super().__init__(lr, weight_decay, criterion)
        self.save_hyperparameters()

        # Embedding layer
        self.features_embedding = FeaturesEmbedding(field_dims, embed_dim)

        # Linear layer
        self.features_linear = FeaturesLinear(field_dims)

        # Pairwise interaction layer
        self.pairwise_interaction = PairwiseInteraction()

    def forward(self, x):
        # Linear regression
        linear = self.features_linear(x.int()) 

        # Pairwise interaction between embeddings
        x_embed = self.features_embedding(x.int())
        pairwise_interaction = self.pairwise_interaction(x_embed)

        # Sum
        y = linear + pairwise_interaction

        return y