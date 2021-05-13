from typing import Union

import torch
import torch.nn.functional as F

from .module import BaseModule
from .sdae import SDAE
from .nmf import NMF
    

def freeze_linear_params(layer, weight_indices, bias_indices = None, dim = 0):
    def freezing_hook_weight_full(grad, weight_multiplier):
        return grad * weight_multiplier.to(grad.device)

    def freezing_hook_bias_full(grad, bias_multiplier):
        return grad * bias_multiplier.to(grad.device)

    weight_multiplier = torch.ones(layer.weight.shape).to(layer.weight.device)
    bias_multiplier = torch.ones(layer.bias.shape).to(layer.bias.device)
    if dim:
        weight_multiplier[:, weight_indices] = 0
    else:
        weight_multiplier[weight_indices] = 0

    if bias_indices:
        bias_multiplier[bias_indices] = 0

    freezing_hook_weight = lambda grad: freezing_hook_weight_full(grad, weight_multiplier)
    freezing_hook_bias = lambda grad: freezing_hook_bias_full(grad, bias_multiplier)

    weight_hook_handle = layer.weight.register_hook(freezing_hook_weight)
    bias_hook_handle = layer.bias.register_hook(freezing_hook_bias)

    return weight_hook_handle, bias_hook_handle

class NMFPP(BaseModule):
    def __init__(self, 
                 user_ae_hparams:dict, 
                 item_ae_hparams:dict, 
                 nmf_hparams:dict,
                 nmf_params:dict = None,
                 alpha:Union[int,float] = 1, 
                 beta:Union[int,float] = 1,
                 lr:float = 1e-3,
                 weight_decay:float = 0,
                 criterion:F = F.mse_loss):
        assert user_ae_hparams["hidden_dims"][-1] and item_ae_hparams["hidden_dims"][-1],\
        "user SDAE and item SDAE latent dimension must be the same"

        super().__init__(lr, weight_decay, criterion)
        self.save_hyperparameters(ignore = ["nmf_params"])
        
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
                    freeze_linear_params(self.nmf.ncf.net.mlp[0], list(range(ncf_input_dim)), dim = 1)


    def forward(self, x, user_x, item_x):
        # Obtain latent factor z
        user_z = self.user_ae.encode(user_x)
        item_z = self.item_ae.encode(item_x)

        # GMF part: concat embeddings and latent factors
        # Concat element wise product of latent factor and embeddings        
        gmf_z_product = user_z * item_z
        gmf_embed_product = self.nmf.gmf.element_wise_product(x)
        gmf_product = torch.cat([gmf_z_product, gmf_embed_product], dim = 1) # concat z 1st then embed

        # NCF part: get output of last hidden layer
        # Concat latent factor and embeddings        
        ncf_z_concat = torch.cat([user_z, item_z], dim = 1)
        ncf_embed_concat = self.nmf.ncf.concatenate(x)
        ncf_concat = torch.cat([ncf_embed_concat, ncf_z_concat], dim = 1) # concat embed 1st then z
        # Feed to net and get output of last hidden layer
        ncf_last_hidden = self.nmf.ncf.net(ncf_concat)

        # Concatenate GMF's element wise product and NCF's last hidden layer output
        concat = torch.cat([gmf_product, ncf_last_hidden], dim = 1)

        # Feed to generalized non-linear layer
        y = self.nmf.net(concat)

        return y

    def backward_loss(self, *batch):
        x, user_x, item_x, y = batch

        loss = 0
        loss += self.alpha * self.criterion(self.user_ae.forward(user_x), user_x)
        loss += self.beta  * self.criterion(self.item_ae.forward(item_x), item_x)
        loss += super().backward_loss(*batch)

        return loss
    



        