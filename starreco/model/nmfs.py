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
                 weight_decay:float = 1e-6,
                 criterion:F = F.mse_loss):
        super().__init__(lr, weight_decay, criterion)
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
        x, user_x, item_x, y = batch

        loss = 0
        loss += self.gmfpp.alpha * self.criterion(self.gmfpp.user_ae.forward(user_x), user_x)
        loss += self.gmfpp.beta  * self.criterion(self.gmfpp.item_ae.forward(item_x), item_x)
        loss += self.ncfpp.alpha * self.criterion(self.ncfpp.user_ae.forward(user_x), user_x)
        loss += self.ncfpp.beta  * self.criterion(self.ncfpp.item_ae.forward(item_x), item_x)
        loss += super().backward_loss(*batch)

        return loss




