import torch
import torch.nn.functional as F

from .module import BaseModule
from .layers import FeaturesEmbedding, FeaturesLinear, PairwiseInteraction

# Done
class AbstractFM(BaseModule):
    """
    Abstract Factorization Machine class.

    - field_dims (list): List of features dimensions. 
    - embed_dim (int): Embedding dimension. 
    - lr (float): Learning rate. 
    - l2_lambda (float): L2 regularization rate. 
    - criterion (F): Criterion or objective or loss function. 

    Warning: This class is an abstract class and should not be used directly.
    """

    def __init__(self, 
                 field_dims:list, 
                 embed_dim:int, 
                 lr:float,
                 weight_decay:float,
                 criterion:F):
        super().__init__(lr, weight_decay, criterion)

        # Embedding layer
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)

        # Linear layer
        self.linear = FeaturesLinear(field_dims)

# Done
class FM(AbstractFM):
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
        super().__init__(field_dims, embed_dim, lr, weight_decay, criterion)
        self.save_hyperparameters()

        # Pairwise interaction
        self.pairwise_interaction = PairwiseInteraction()

    def forward(self, x):
        # Generate embeddings
        embed_x = self.embedding(x.int())
        
        # Prediction
        linear = self.linear(x.int()) 
        pairwise_interaction = self.pairwise_interaction(embed_x)
        y = linear + pairwise_interaction

        return y