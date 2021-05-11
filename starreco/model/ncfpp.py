from typing import Union

import torch
import torch.nn.functional as F

from .module import BaseModule
from .layer import FeaturesEmbedding, MultilayerPerceptrons
from .sdae import SDAE
from .ncf import NCF


class NCFPP(BaseModule):
    def __init__(self, 
                 user_ae_hparams:dict, 
                 item_ae_hparams:dict, 
                 ncf_hparams:dict,
                 ncf_params:dict = None,
                 alpha:Union[int,float] = 1, 
                 beta:Union[int,float] = 1,
                 lr:float = 1e-3,
                 weight_decay:float = 1e-6,
                 criterion:F = F.mse_loss):
        assert user_ae_hparams["hidden_dims"][-1] and item_ae_hparams["hidden_dims"][-1],\
        "user SDAE and item SDAE latent dimension must be the same"

        super().__init__(lr, weight_decay, criterion)
        self.save_hyperparameters(ignore = ["ncf_params"])
        
        self.alpha = alpha
        self.beta = beta

        self.user_ae = SDAE(**user_ae_hparams)
        self.item_ae = SDAE(**item_ae_hparams)
        self.ncf = NCF(**ncf_hparams)

        if ncf_params:
            self.ncf.load_state_dict(ncf_params)
            input_weights = self.ncf.net.mlp[0].weight.clone()

        # Replace the first layer with reshape input features
        latent_dim = user_ae_hparams["hidden_dims"][-1] and item_ae_hparams["hidden_dims"][-1]
        input_dim = self.ncf.net.mlp[0].in_features
        output_dim = self.ncf.net.mlp[0].out_features
        self.ncf.net.mlp[0] = torch.nn.Linear(input_dim + latent_dim * 2, output_dim)

        if ncf_params:
            with torch.no_grad():
                self.ncf.net.mlp[0].weight[:, :input_dim] = input_weights

    def encode_concatenate(self, x, user_x, item_x):
        # Obtain latent factor z
        user_z = self.user_ae.encode(user_x)
        item_z = self.item_ae.encode(item_x)

        # Concat latent factor and embeddings        
        z_concat = torch.cat([user_z, item_z], dim = 1)
        embed_concat = self.ncf.concatenate(x)

        # Concat  embeddings and latent factors
        concat = torch.cat([embed_concat, z_concat], dim = 1)

        return concat

    def forward(self, x, user_x, item_x):
        # Element wise product between user and items embeddings
        concat = self.encode_concatenate(x, user_x, item_x)
        
        # Feed element wise product to generalized non-linear layer
        y = self.ncf.net(concat)

        return y

    def backward_loss(self, *batch):
        x, user_x, item_x, y = batch

        loss = 0
        loss += self.alpha * self.criterion(self.user_ae.forward(user_x), user_x)
        loss += self.beta  * self.criterion(self.item_ae.forward(item_x), item_x)
        loss += super().backward_loss(*batch)

        return loss