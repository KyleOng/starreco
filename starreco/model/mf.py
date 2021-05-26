import torch
import torch.nn.functional as F

from .module import BaseModule
from .layers import FeaturesEmbedding
from .mixins import FeaturesEmbeddingMixin 

# Done
class MF(BaseModule, FeaturesEmbeddingMixin):
    """
    Matrix Factorization.

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
                 l2_lambda:float = 1e-6,
                 criterion:F = F.mse_loss):
        assert len(field_dims) == 2, "`field_dims` should contains only 2 elements (user and item field)"

        super().__init__(lr, l2_lambda, criterion)
        self.save_hyperparameters()

        # Embedding layer
        self.features_embedding = FeaturesEmbedding(field_dims, embed_dim)

    def forward(self, x):
        # Generate embeddings
        user_embed, item_embed = self.user_item_embeddings(x)

        # Dot product between user and items embeddings
        dot = torch.sum(user_embed * item_embed, dim = 1)
        
        # Prediction and reshape to match target shape
        y = dot.view(dot.shape[0], -1)

        return y