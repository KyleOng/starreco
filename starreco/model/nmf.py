from typing import Union
import copy

import numpy as np
import torch
import torch.nn.functional as F

from .module import BaseModule
from .layer import FeaturesEmbedding, MultilayerPerceptrons, ActivationFunction
from .gmf import GMFmodel
from .ncf import NCFmodel

class NMFmodel(torch.nn.Module):
    """
    Neural Matrix Factorization
    """
    def __init__(self, 
                 gmf:GMFmodel,
                 ncf:NCFmodel,
                 freeze_pretrain:bool):
        super().__init__()
        
        # Clone pretrained models
        self.gmf = copy.deepcopy(gmf)
        self.ncf = copy.deepcopy(ncf)

        # Freeze model
        if freeze_pretrain:
            for param in self.gmf.parameters():
                param.requires_grad = False
            for param in self.ncf.parameters():
                param.requires_grad = False

        # Remove GMF output layer
        del self.gmf.net

        # Remove NCF output layer
        del self.ncf.net.mlp[-1]
        if type(self.ncf.net.mlp[-1]) == torch.nn.Linear:
            del self.ncf.net.mlp[-1]

        # Get input dim 
        input_dim = self.gmf.features_embedding.embedding.embedding_dim 
        for i in range(0, len(self.ncf.net.mlp))[::-1]:
            if type(self.ncf.net.mlp[i]) == torch.nn.Linear:
                input_dim += self.ncf.net.mlp[i].out_features
                break
        # Hybrid network
        self.net = MultilayerPerceptrons(input_dim = input_dim, 
                                         output_layer = "relu")
    
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
    

class NMF(BaseModule):
    """
    Neural Matrix Factorization
    """
    def __init__(self, 
                 gmf:GMFmodel,
                 ncf:NCFmodel,
                 freeze_pretrain:bool = True,
                 lr:float = 1e-3,
                 weight_decay:float = 1e-6,
                 criterion:F = F.mse_loss):
        
        super().__init__(lr, weight_decay, criterion)
        self.model = NMFmodel(gmf, ncf, freeze_pretrain)
        self.save_hyperparameters()

    def forward(self, x):
        return self.model.forward(x)



        