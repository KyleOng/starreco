import torch.nn.functional as F

from .lr import LR
from .layers import FeaturesEmbedding, PairwiseInteraction

# Done
class FM(LR):
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
        super().__init__(field_dims, lr, weight_decay, criterion)
        self.save_hyperparameters()

        # Embedding layer
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)

        # Pairwise interaction
        self.pairwise_interaction = PairwiseInteraction()

    def forward(self, x):
        # Generate embeddings
        embed_x = self.embedding(x.int())
        
        # Prediction
        linear = super().forward(x)
        pairwise_interaction = self.pairwise_interaction(embed_x)
        y = linear + pairwise_interaction

        return y