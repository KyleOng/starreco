from typing import Union
import copy

import numpy as np
import torch
import torch.nn.functional as F

from starreco.model import (FeaturesEmbedding, 
                            MultilayerPerceptrons, 
                            Module)
from .gmf import GMF
from .ncf import NCF

class NMF(Module):
    """
    Generalized Matrix Factorization
    """
    def __init__(self, gmf_params:dict, ncf_params:dict,
                 gmf_state_dict:dict = None,
                 ncf_state_dict:dict = None,
                 pretrain:bool = False,
                 lr:float = 1e-2,
                 weight_decay:float = 1e-6,
                 criterion:F = F.mse_loss,
                 save_hyperparameters:bool = True):
        """
        Hyperparameters setting.

        :param embed_dim (int): Embeddings dimensions. Default: 8

        :param activation (str): Activation Function. Default: "relu".

        :param lr (float): Learning rate. Default: 1e-3

        :param weight_decay (float): L2 regularization weight decay: Default: 1e-3

        :param criterion (F): Objective function. Default: F.mse_loss
        """
        super().__init__(lr, weight_decay, criterion)
        gmf_params["save_hyperparameters"] = False
        self.gmf = GMF(**gmf_params)
        if gmf_state_dict:
            self.gmf.load_state_dict(gmf_state_dict)

        ncf_params["save_hyperparameters"] = False
        self.ncf = NCF(**ncf_params)
        if ncf_state_dict:
            self.ncf.load_state_dict(ncf_state_dict)

        # Freeze embedding weights
        if pretrain:
            self.gmf.embedding.embedding.weight.requires_grad = False
            self.ncf.embedding.embedding.weight.requires_grad = False

        # Remove GMF output layer
        self.gmf.nn = None

        # Add GMF embedding dim to `input_dim`
        input_dim = self.gmf.embedding.embedding.weight.shape[1]

        # Remove NCF output layer and freeze hidden weights
        ncf_nn_blocks = []
        for i, module in enumerate(self.ncf.nn.mlp):
            # Freeze weights and bias if pretrain
            if hasattr(self.ncf.nn.mlp[i], "weight") and pretrain:
                self.ncf.nn.mlp[i].weight.requires_grad = False
            if hasattr(self.ncf.nn.mlp[i], "bias"):
                self.ncf.nn.mlp[i].bias.requires_grad = False
            if type(module) == torch.nn.Linear:
                if module.out_features == 1:
                    # Add NCF last hidden input dim to `input_dim`
                    input_dim += module.in_features 
                    break
            ncf_nn_blocks.append(module)
        self.ncf.nn = torch.nn.Sequential(*ncf_nn_blocks)
        
        # Singlelayer perceptrons
        # input_dim = GMF embedding dim + NCF last hidden input dim
        self.nn = MultilayerPerceptrons(input_dim = input_dim, 
                                        output_layer = "relu")

        # Save hyperparameters to checkpoint
        if save_hyperparameters:
            self.save_hyperparameters()
    
    def forward(self, x):
        """
        Perform operations.

        :x (torch.tensor): Input tensors of shape (batch_size, 2) user and item.

        :return (torch.tensor): Output prediction tensors of shape (batch_size, 1)
        """
        gmf_embedding = self.gmf.embedding(x)
        gmf_user_embedding = gmf_embedding[:, 0]
        gmf_item_embedding = gmf_embedding[:, 1]
        gmf_x = gmf_user_embedding *  gmf_item_embedding 
        
        ncf_x = self.ncf(x)

        concat = torch.cat([gmf_x, ncf_x], dim = 1)
        y = self.nn(concat)

        return y