from typing import Union
import copy

import torch
import torch.nn.functional as F

from .module import BaseModule
from .layer import FeaturesEmbedding, MultilayerPerceptrons
from .gmfpp import GMFPP
from .ncfpp import NCFPP


class NMFS(BaseModule):
    """
    Neural Matrix Factorization Sharp or # or ++++
    """
    def __init__(self, 
                 gmfpp_hparams:dict,
                 ncfpp_hparams:dict,
                 gmfpp_params:dict = None,
                 ncfpp_params:dict = None,
                 freeze_pretrain:bool = True,
                 lr:float = 1e-3,
                 l2_lambda:float = 1e-3,
                 criterion:F = F.mse_loss):
        if freeze_pretrain:
            super().__init__(lr, l2_lambda, criterion)
        else:
            super().__init__(lr, 0, criterion)
            self.l2_lambda = l2_lambda
        self.save_hyperparameters(ignore = ["gmfpp_params", "ncfpp_params"])

        self.gmfpp = GMFPP(**gmfpp_hparams)
        self.ncfpp = NCFPP(**ncfpp_hparams)

        # Load pretrained weights
        if gmfpp_params:
            self.gmfpp.load_state_dict(gmfpp_params)
        if ncfpp_params:
            self.ncfpp.load_state_dict(ncfpp_params)

        # Freeze model
        if freeze_pretrain:
            self.gmfpp.freeze()
            self.ncfpp.freeze()

        # Remove GMF output layer
        del self.gmfpp.gmf.net

        # Remove NCF output layer
        del self.ncfpp.ncf.net.mlp[-1]
        if type(self.ncfpp.ncf.net.mlp[-1]) == torch.nn.Linear:
            del self.ncfpp.ncf.net.mlp[-1]

        # Add input dim
        input_dim = gmfpp_hparams["gmf_hparams"]["embed_dim"]
        input_dim += gmfpp_hparams["user_ae_hparams"]["hidden_dims"][-1] and gmfpp_hparams["item_ae_hparams"]["hidden_dims"][-1]
        input_dim += ncfpp_hparams["ncf_hparams"]["hidden_dims"][-1]
        self.net = MultilayerPerceptrons(input_dim = input_dim, 
                                         output_layer = "relu")
        self.freeze_pretrain = freeze_pretrain

    def forward(self, x, user, item):
        # GMFPP part: Element wise product between latent factor and embeddings
        gmfpp_product = self.gmfpp.encode_element_wise_product(x, user, item)

        # NCFPP part: Get output of last hidden layer
        ncfpp_last_hidden = self.ncfpp.forward(x, user ,item)

        # Concatenate GMFPP's element wise product and NCFPP's last hidden layer output
        concat = torch.cat([gmfpp_product, ncfpp_last_hidden], dim = 1)

        # Feed to generalized non-linear layer
        y = self.net(concat)

        return y

    def backward_loss(self, *batch):
        if self.freeze_pretrain:
            return super().backward_loss(*batch)
        else:
            x, user_x, item_x, y = batch

            gmfpp_user_loss = self.criterion(self.gmfpp.user_ae.forward(user_x), user_x)
            gmfpp_user_l2_reg = torch.tensor(0.).to(self.device)
            for param in self.gmfpp.user_ae.parameters():
                gmfpp_user_l2_reg += torch.norm(param).to(self.device)
            gmfpp_user_loss += self.l2_lambda * gmfpp_user_l2_reg

            gmfpp_item_loss = self.criterion(self.gmfpp.item_ae.forward(item_x), item_x)
            gmfpp_item_l2_reg = torch.tensor(0.).to(self.device)
            for param in self.gmfpp.item_ae.parameters():
                gmfpp_item_l2_reg += torch.norm(param).to(self.device)
            gmfpp_item_loss += self.l2_lambda * gmfpp_item_l2_reg

            ncfpp_user_loss = self.criterion(self.ncfpp.user_ae.forward(user_x), user_x)
            ncfpp_user_l2_reg = torch.tensor(0.).to(self.device)
            for param in self.ncfpp.user_ae.parameters():
                ncfpp_user_l2_reg += torch.norm(param).to(self.device)
            ncfpp_user_loss += self.l2_lambda * ncfpp_user_l2_reg

            ncfpp_item_loss = self.criterion(self.ncfpp.item_ae.forward(item_x), item_x)
            ncfpp_item_l2_reg = torch.tensor(0.).to(self.device)
            for param in self.ncfpp.item_ae.parameters():
                ncfpp_item_l2_reg += torch.norm(param).to(self.device)
            ncfpp_item_loss += self.l2_lambda * ncfpp_item_l2_reg

            cf_loss = super().backward_loss(*batch)
            cf_l2_reg = torch.tensor(0.).to(self.device)
            for param in self.gmfpp.gmf.parameters():
                cf_l2_reg += torch.norm(param).to(self.device)
            for param in self.ncfpp.ncf.parameters():
                cf_l2_reg += torch.norm(param).to(self.device)
            for param in self.net.parameters():
                cf_l2_reg += torch.norm(param).to(self.device)
            cf_loss += self.l2_lambda * cf_l2_reg

            loss = cf_loss + self.gmfpp.alpha * gmfpp_user_loss + self.gmfpp.beta * gmfpp_item_loss + self.ncfpp.alpha * ncfpp_user_loss + self.ncfpp.beta  * ncfpp_item_loss

            return loss




