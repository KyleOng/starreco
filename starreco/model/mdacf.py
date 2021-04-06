import torch
import torch.nn.functional as F

from starreco.model import (Module)
from .mda import mDA

class mDACF(Module):
    def __init__(self, user_lookup:torch.Tensor, item_lookup:torch.Tensor, user_mda:mDA, item_mda:mDA, 
                 lr:float = 1e-3,
                 weight_decay:float = 0,
                 criterion:F = F.mse_loss,
                 save_hyperparameters:bool = True):

        super().__init__(lr, weight_decay, criterion)
        self.user_lookup = user_lookup
        self.item_lookup = item_lookup

        # Use mDA encoder weights for weights update (learning).
        # Freeze mDA decoder weights and bias.
        # User latent features
        self.user_mda = user_mda
        #self.user_dim = self.user_mda.encoder[0].weight.shape[0]
        self.user_mda.decoder[0].weight.requires_grad = False
        self.user_mda.decoder[0].bias.requires_grad = False
        # Item latent features
        self.item_mda = item_mda
        #self.item_dim = self.item_mda.encoder[0].weight.shape[0]
        self.item_mda.decoder[0].weight.requires_grad = False
        self.item_mda.decoder[0].bias.requires_grad = False

    def forward(self, x):
        # Device is determined during training
        user_x = torch.index_select(self.user_lookup.to(self.device), 0, x[:, 0])
        item_x = torch.index_select(self.item_lookup.to(self.device), 0, x[:, 1])

        user_latent = self.user_mda.encode(user_x)
        item_latent = self.item_mda.encode(item_x)

        # Dot product between embeddings
        matrix = torch.mm(user_latent, item_latent.T)

        # Obtain diagonal part of the outer product result
        diagonal = matrix.diag()

        # Reshape to match evaluation shape
        return diagonal.view(diagonal.shape[0], -1)        
