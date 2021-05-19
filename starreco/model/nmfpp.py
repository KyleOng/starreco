from typing import Union

import torch
import torch.nn.functional as F

from .module import BaseModule
from .sdae import SDAE
from .nmf import NMF
from .utils import freeze_partial_linear_params

class NMFPP(BaseModule):
    def __init__(self, 
                 user_ae_hparams:dict, 
                 item_ae_hparams:dict, 
                 nmf_hparams:dict,
                 nmf_params:dict = None,
                 alpha:Union[int,float] = 1, 
                 beta:Union[int,float] = 1,
                 lr:float = 1e-3,
                 l2_lambda:float = 0,
                 criterion:F = F.mse_loss):
        assert user_ae_hparams["hidden_dims"][-1] and item_ae_hparams["hidden_dims"][-1],\
        "user SDAE and item SDAE latent dimension must be the same"

        super().__init__(lr, 0, criterion)
        self.save_hyperparameters(ignore = ["nmf_params"])
        
        self.l2_lambda = l2_lambda
        self.alpha = alpha
        self.beta = beta

        self.user_ae = SDAE(**user_ae_hparams)
        self.item_ae = SDAE(**item_ae_hparams)
        self.nmf = NMF(**nmf_hparams)

        if nmf_params:
            self.nmf.load_state_dict(nmf_params)
            ncf_input_weights = self.nmf.ncf.net.mlp[0].weight.clone()
            nmf_input_weights = self.nmf.net.mlp[0].weight.clone()

        # Replace the first layer with reshape input features
        latent_dim = user_ae_hparams["hidden_dims"][-1] and item_ae_hparams["hidden_dims"][-1]

        ncf_input_dim = self.nmf.ncf.net.mlp[0].in_features
        ncf_output_dim = self.nmf.ncf.net.mlp[0].out_features
        self.nmf.ncf.net.mlp[0] = torch.nn.Linear(ncf_input_dim + latent_dim * 2, ncf_output_dim)

        nmf_input_dim = self.nmf.net.mlp[0].in_features
        nmf_output_dim = self.nmf.net.mlp[0].out_features
        self.nmf.net.mlp[0] = torch.nn.Linear(nmf_input_dim + latent_dim, nmf_output_dim)

        if nmf_params:
            with torch.no_grad():
                self.nmf.net.mlp[0].weight[:, -nmf_input_dim:] = nmf_input_weights
                self.nmf.ncf.net.mlp[0].weight[:, :ncf_input_dim] = ncf_input_weights
                if nmf_hparams["freeze_pretrain"]:
                    freeze_partial_linear_params(self.nmf.ncf.net.mlp[0], list(range(ncf_input_dim)), dim = 1)

    def gmf_element_wise_product(self, x, user_z, item_z):
        # Obtain product of latent factor and embeddings        
        z_product = user_z * item_z
        embed_product = self.nmf.gmf.element_wise_product(x)

        # Concat  embeddings and latent factors
        product = torch.cat([z_product, embed_product], dim = 1)

        return product
    
    def ncf_forward(self, x, user_z, item_z):
        # Concat latent factor and embeddings        
        z_concat = torch.cat([user_z, item_z], dim = 1)
        embed_concat = self.nmf.ncf.concatenate(x)

        # Concat embeddings and latent factors
        concat = torch.cat([embed_concat, z_concat], dim = 1)

        # Get output of last hidden layer
        return self.nmf.ncf.net(concat)

    def forward(self, x, user_x, item_x):
        # Obtain latent factor z
        user_z = self.user_ae.encode(user_x)
        item_z = self.item_ae.encode(item_x)

        # GMF part: Element wise product between embeddings
        gmf_product = self.gmf_element_wise_product(x, user_z, item_z)

        # NCF part: Concatenate embedding and latent vector and get output of last hidden layer
        ncf_last_hidden = self.ncf_forward(x, user_z, item_z)

        # Concatenate GMF's element wise product and NCF's last hidden layer output
        concat = torch.cat([gmf_product, ncf_last_hidden], dim = 1)

        # Feed to generalized non-linear layer
        y = self.nmf.net(concat)

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
        for param in self.nmf.parameters():
            cf_l2_reg += torch.norm(param).to(self.device)
        cf_loss += self.l2_lambda * cf_l2_reg

        loss = cf_loss + self.alpha * user_loss + self.beta * item_loss

        return loss
    



        