import torch
import torch.nn.functional as F

from .module import BaseModule
from .layer import FeaturesEmbedding

class MF(torch.nn.Module):
    """
    Matrix Factorization
    """
    def __init__(self, field_dims:list, 
                 embed_dim:int = 8):
        super().__init__()

        # Embedding layer
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)

    def forward(self, x):
        # Generate embeddings
        x_embed = self.embedding(x)
        # Seperate user (1st column) and item (2nd column) embeddings from generated embeddings
        user_embed = x_embed[:, 0]
        item_embed = x_embed[:, 1]

        # Dot product between user and items embeddings
        dot = torch.mm(user_embed, item_embed.T)
        # Obtain diagonal part of the dot product result
        diagonal = dot.diag()
        # Reshape to match output shape
        y = diagonal.view(diagonal.shape[0], -1)

        return y

class MFModule(BaseModule):
    def __init__(self, field_dims:list, 
                 embed_dim:int = 8, 
                 lr:float = 1e-3,
                 weight_decay:float = 1e-6,
                 criterion:F = F.mse_loss):
        super().__init__(lr, weight_decay, criterion)
        self.model = MF(field_dims, embed_dim)
        self.save_hyperparameters()
    

