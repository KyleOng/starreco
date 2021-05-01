import torch
import torch.nn.functional as F

from .module import BaseModule
from .layer import FeaturesEmbedding

class MFmodel(torch.nn.Module):
    """
    Matrix Factorization model
    """

    def __init__(self, field_dims:list, 
                 embed_dim:int):
        super().__init__()

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
        
        # Reshape to match target shape
        y = dot.view(dot.shape[0], -1)

        return y


class MF(BaseModule):
    """
    Matrix Factorization lightning module
    """

    def __init__(self, field_dims:list, 
                 embed_dim:int = 8, 
                 lr:float = 1e-3,
                 weight_decay:float = 1e-6,
                 criterion:F = F.mse_loss):
        super().__init__(lr, weight_decay, criterion)
        self.model = MFmodel(field_dims, embed_dim)
        self.save_hyperparameters()

    def forward(self, x):
        return self.model.forward(x)