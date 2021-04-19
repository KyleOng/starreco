from typing import Union
import copy

import numpy as np
import torch
import torch.nn.functional as F

from .module import BaseModule
from .layer import FeaturesEmbedding, MultilayerPerceptrons
from .gmf import GMF
from .ncf import NCF

class NMF(torch.nn.Module):
    """
    Neural Matrix Factorization
    """
    def __init__(self, gmf, ncf,
                 pretrain:bool = False):
        super().__init__()
        self.gmf = copy.deepcopy(gmf)
        self.ncf = copy.deepcopy(ncf)

        # Freeze embedding weights if pretrain
        if pretrain:
            self.gmf.embedding.embedding.weight.requires_grad = False
            self.ncf.embedding.embedding.weight.requires_grad = False

        # Remove GMF output layer
        self.gmf.slp = None

        # Add GMF embedding dim to `input_dim`
        input_dim = self.gmf.embedding.embedding.weight.shape[1]

        # Remove NCF output layer and freeze hidden weights and biases
        ncf_mlp_blocks = []
        for i, module in enumerate(self.ncf.mlp.mlp):
            # Freeze weights and bias if pretrain
            if hasattr(self.ncf.mlp.mlp[i], "weight") and pretrain:
                self.ncf.mlp.mlp[i].weight.requires_grad = False
            if hasattr(self.ncf.mlp.mlp[i], "bias") and pretrain:
                self.ncf.mlp.mlp[i].bias.requires_grad = False
            if type(module) == torch.nn.Linear:
                if module.out_features == 1:
                    # Add NCF last hidden input dim to `input_dim`
                    input_dim += module.in_features 
                    break
            ncf_mlp_blocks.append(module)
        self.ncf.mlp = torch.nn.Sequential(*ncf_mlp_blocks)
        
        # Singlelayer perceptrons
        # input_dim = GMF embedding dim + NCF last hidden input dim
        self.slp = MultilayerPerceptrons(input_dim = input_dim, 
                                         output_layer = "relu")
    
    def forward(self, x):
        # GMF part
        gmf_x_embed = self.gmf.embedding(x)
        gmf_user_embed = gmf_x_embed[:, 0]
        gmf_item_embed = gmf_x_embed[:, 1]
        # Element wise product between embeddings
        gmf_product = gmf_user_embed *  gmf_item_embed

        # NCF part
        # Get output of last hidden layer
        ncf_last_hidden = self.ncf(x)

        # Concatenate GMF's element wise product and NCF's last hidden layer output
        concat = torch.cat([gmf_product, ncf_last_hidden], dim = 1)

        # Feed to generalized non-linear layer
        y = self.slp(concat)

        return y

class NMFModule(BaseModule):
    def __init__(self, gmf:GMF, ncf:NCF,
                 pretrain:bool = False,
                 lr:float = 1e-3,
                 weight_decay:float = 1e-6,
                 criterion:F = F.mse_loss,
                 save_hyperparameters:bool = True):
        super().__init__(lr, weight_decay, criterion)
        self.model = NMF(gmf, ncf, pretrain)
        self.save_hyperparameters()
    
        