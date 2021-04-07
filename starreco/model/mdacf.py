import torch
import torch.nn.functional as F

from starreco.model import (Module)
from .mda import mDA

class mDACF(Module):
    def __init__(self, user_mda:mDA, item_mda:mDA, feature_dims:list, 
                 lr:float = 1e-3,
                 weight_decay:float = 0,
                 criterion:F = F.mse_loss,
                 save_hyperparameters:bool = True):
        super().__init__(lr, weight_decay, criterion)
        self.user_mda = user_mda
        self.item_mda = item_mda
        self.feature_dims = feature_dims

    def forward(self, x):
        user_x = x[:, :feature_dims[0]]
        item_x = x[:, -feature_dims[1]:]

        user_z = self.user_mda.encode(user_x)
        item_z = self.item_mda.encode(item_x)

        user_y = self.user_mda.decoder(user_z)
        item_y = self.item_mda.decoder(item_z)

        # Dot product between embeddings
        dot = torch.mm(user_z, item_z.T)

        # Obtain diagonal part of the outer product result
        diagonal = dot.diag()

        # Reshape to match evaluation shape
        y = diagonal.view(diagonal.shape[0], -1)        

        return y

    def evaluate(self, x, y):
        user_x = x[:, :feature_dims[0]]
        item_x = x[:, -feature_dims[1]:]

        user_loss = self.user_mda.evaluate(user_x)
        item_loss = self.item_mda.evaluate(item_x)

        loss = user_loss + item_loss + super().evaluate(x, y)

        return loss
