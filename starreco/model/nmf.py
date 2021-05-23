from typing import Union

import numpy as np
import torch
import torch.nn.functional as F

from .module import BaseModule
from .layer import FeaturesEmbedding, MultilayerPerceptrons
from .gmf import GMF
from .ncf import NCF
    
# Done
class NMF(BaseModule):
    """
    Neural Matrix Factorization
    """
    def __init__(self, 
                 gmf_hparams:dict,
                 ncf_hparams:dict,
                 gmf_params:dict = None,
                 ncf_params:dict = None,
                 freeze_pretrain:bool = True,
                 lr:float = 1e-3,
                 l2_lambda:float = 1e-3,
                 criterion:F = F.mse_loss):
        super().__init__(lr, l2_lambda, criterion)
        self.save_hyperparameters(ignore = ["gmf_params", "ncf_params"])
        
        self.gmf = GMF(**gmf_hparams)
        self.ncf = NCF(**ncf_hparams)

        # Load pretrained weights
        if gmf_params:
            self.gmf.load_state_dict(gmf_params)
        if ncf_params:
            self.ncf.load_state_dict(ncf_params)

        # Freeze model
        if freeze_pretrain:
            self.gmf.freeze()
            self.ncf.freeze()

        # Remove GMF output layer
        del self.gmf.net

        # Remove NCF output layer
        del self.ncf.net.mlp[-1]
        if type(self.ncf.net.mlp[-1]) == torch.nn.Linear:
            del self.ncf.net.mlp[-1]

        # Add input dim
        input_dim = gmf_hparams["embed_dim"] + ncf_hparams["hidden_dims"][-1]
        self.net = MultilayerPerceptrons(input_dim = input_dim, output_layer = "relu")

    def forward(self, x):
        # GMF part: Element wise product between embeddings
        gmf_product = self.gmf.element_wise_product(x)

        # NCF part: Get output of last hidden layer
        ncf_last_hidden = self.ncf.forward(x)

        # Concatenate GMF's element wise product and NCF's last hidden layer output
        concat = torch.cat([gmf_product, ncf_last_hidden], dim = 1)

        # Feed to generalized non-linear layer
        y = self.net(concat)

        return y
    



        