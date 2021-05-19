from typing import Union

import torch
import torch.nn.functional as F

from .module import BaseModule
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
                 l2_lambda:float = 1e-3,
                 criterion:F = F.mse_loss):
        assert user_ae_hparams["hidden_dims"][-1] and item_ae_hparams["hidden_dims"][-1],\
        "user SDAE and item SDAE latent dimension must be the same"

        super().__init__(lr, 0, criterion)
        self.save_hyperparameters(ignore = ["ncf_params"])
        
        self.l2_lambda = l2_lambda
        self.alpha = alpha
        self.beta = beta

        self.user_ae = SDAE(**user_ae_hparams)
        self.item_ae = SDAE(**item_ae_hparams)
        self.ncf = NCF(**ncf_hparams)

        if ncf_params:
            self.ncf.load_state_dict(ncf_params)

        # Replace the first layer with reshape input features
        latent_dim = user_ae_hparams["hidden_dims"][-1] and item_ae_hparams["hidden_dims"][-1]
        input_dim = self.ncf.net.mlp[0].in_features
        output_dim = self.ncf.net.mlp[0].out_features
        # Obtain first layer input weight before reshaping
        input_weights = self.ncf.net.mlp[0].weight.clone()
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

        # Concat embeddings and latent factors
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

        user_loss = self.criterion(self.user_ae.forward(user_x), user_x)
        user_l2_reg = torch.tensor(0.).to(self.device)
        for param in self.user_ae.parameters():
            user_l2_reg += torch.norm(param).to(self.device)
        user_loss += self.l2_lambda * user_l2_reg

        item_loss = self.criterion(self.item_ae.forward(item_x), item_x)
        item_l2_reg = torch.tensor(0.).to(self.device)
        for param in self.item_ae.parameters():
            item_l2_reg += torch.norm(param).to(self.device)
        item_loss += self.l2_lambda * item_l2_reg

        cf_loss = super().backward_loss(*batch)
        cf_l2_reg = torch.tensor(0.).to(self.device)
        for param in self.ncf.parameters():
            cf_l2_reg += torch.norm(param).to(self.device)
        cf_loss += self.l2_lambda * cf_l2_reg

        loss = cf_loss + self.alpha * user_loss + self.beta * item_loss

        return loss