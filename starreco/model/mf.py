import torch
import torch.nn.functional as F

from .module import BaseModule
from .layer import FeaturesEmbedding

# Done
class MF(BaseModule):
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
        super().__init__(lr, l2_lambda, criterion)
        self.save_hyperparameters()

        # Embedding layer
        self.features_embedding = FeaturesEmbedding(field_dims, embed_dim)

    def forward(self, x):
        # Generate embeddings
        x_embed = self.features_embedding(x.int())
        # Seperate user (1st column) and item (2nd column) embeddings from generated embeddings
        user_embed = x_embed[:, 0]
        item_embed = x_embed[:, 1]

        # Dot product between user and items embeddings
        dot = torch.sum(user_embed * item_embed, dim = 1)
        
        # Prediction and reshape to match target shape
        y = dot.view(dot.shape[0], -1)

        return y