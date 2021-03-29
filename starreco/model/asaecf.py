from typing import Union

import torch
import torch.nn.functional as F

from starreco.model import (Module)
from .aae import AAE
from .asae import ASAE

class ASAECF(Module):
    def __init__(self, user_lookup:torch.Tensor, item_lookup:torch.Tensor, user_aae:Union[AAE, ASAE], item_aae:Union[AAE, ASAE],
                 lr:float = 1e-3,
                 weight_decay:float = 0,
                 criterion:F = F.mse_loss):
        super().__init__(lr, weight_decay, criterion)
        self.user_lookup = user_lookup
        self.item_lookup = item_lookup

        self.user_aae = user_aae
        for i, module in enumerate(self.user_aae.decoder.mlp):
            if type(module) == torch.nn.Linear:
                try:
                    latent_dim
                except:
                    latent_dim = self.user_aae.decoder.mlp[i].weight.shape[1]
                else:
                    pass
                self.user_aae.decoder.mlp[i].weight.requires_grad = False
                self.user_aae.decoder.mlp[i].bias.requires_grad = False

        self.item_aae = item_aae
        for i, module in enumerate(self.item_aae.decoder.mlp):
            if type(module) == torch.nn.Linear:
                try:
                    latent_dim
                except:
                    latent_dim = self.item_aae.decoder.mlp[i].weight.shape[1]
                else:
                    pass
                self.item_aae.decoder.mlp[i].weight.requires_grad = False
                self.item_aae.decoder.mlp[i].bias.requires_grad = False

    def forward(self, x):
        # Device is determined during training
        user_x = torch.index_select(self.user_lookup.to(self.device), 0, x[:, 0])
        item_x = torch.index_select(self.item_lookup.to(self.device), 0, x[:, 1])

        user_feature = user_x[:, :self.user_aae.feature_dim]
        user_x = user_x[:, self.user_aae.feature_dim:]

        item_feature = item_x[:, :self.item_aae.feature_dim]
        item_x = item_x[:, self.item_aae.feature_dim:]

        user_latent = self.user_aae.encode(user_x, user_feature)
        item_latent = self.item_aae.encode(item_x, item_feature)

        # Dot product between embeddings
        matrix = torch.mm(user_latent, item_latent.T)

        # Obtain diagonal part of the outer product result
        diagonal = matrix.diag()

        # Reshape to match evaluation shape
        return diagonal.view(diagonal.shape[0], -1)        
