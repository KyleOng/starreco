import torch
import torch.nn.functional as F

from .module import BaseModule
from .layers import FeaturesEmbedding

# Done
class MF(BaseModule):
    """
    Matrix Factorization.

    - field_dims (list): List of features dimensions.
    - embed_dim (int): Embedding dimension. Default: 8.
    - lr (float): Learning rate. Default: 1e-3.
    - weight_decay (float): L2 regularization rate. Default: 1e-3.
    - criterion: Criterion or objective or loss function. Default: F.mse_loss.
    """
    
    def __init__(self, 
                 field_dims:list, 
                 embed_dim:int = 8, 
                 lr:float = 1e-3,
                 weight_decay:float = 1e-6,
                 criterion = F.mse_loss):
        assert len(field_dims) == 2, "`field_dims` should contains only 2 elements (user and item field)"

        super().__init__(lr, weight_decay, criterion)
        self.save_hyperparameters()

        # Embedding layer
        self.features_embedding = FeaturesEmbedding(field_dims, embed_dim)

    def forward(self, x):
        # Dot product between user and item embeddings
        x_embed = self.features_embedding(x.int())
        user_embed, item_embed = x_embed[:, 0], x_embed[:, 1]
        dot = torch.sum(user_embed * item_embed, dim = 1)
        
        # Reshape to match target shape
        y = dot.view(dot.shape[0], -1)

        return y