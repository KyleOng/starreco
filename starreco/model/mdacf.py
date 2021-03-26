import torch
import torch.nn.functional as F

from starreco.model import (Module)
from .mda import mDA

class mDACF(Module):
    def __init__(self, user_mda:mDA, item_mda:mDA,
                 lr:float = 1e-3,
                 weight_decay:float = 0,
                 criterion:F = F.mse_loss):

        super().__init__(lr, weight_decay, criterion)

        # Use mDA encoder weights for weights update (learning).
        # Freeze mDA decoder weights and bias.
        # User latent features
        self.user_mda = user_mda
        self.user_dim = self.user_mda.encoder[0].weight.shape[0]
        self.user_mda.decoder[0].weight.requires_grad = False
        self.user_mda.decoder[0].bias.requires_grad = False
        # Item latent features
        self.item_mda = item_mda
        self.item_dim = self.item_mda.encoder[0].weight.shape[0]
        self.item_mda.decoder[0].weight.requires_grad = False
        self.item_mda.decoder[0].bias.requires_grad = False

    def forward(self, x):
        user_embedding = self.user_mda.encode(x[:, :self.user_dim])
        item_embedding = self.item_mda.encode(x[:, -self.item_dim:])

        # Dot product between embeddings
        matrix = torch.mm(user_embedding, item_embedding.T)

        # Obtain diagonal part of the outer product result
        diagonal = matrix.diag()

        # Reshape to match evaluation shape
        return diagonal.view(diagonal.shape[0], -1)        
