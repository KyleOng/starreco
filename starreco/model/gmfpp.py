from typing import Union
import copy

import torch
import torch.nn.functional as F

from .module import BaseModule
from .layer import FeaturesEmbedding, MultilayerPerceptrons
from .sdae import SDAEmodel

class GMFPPmodel(torch.nn.Module):
    """
    Generalized Matrix Factorization ++ model
    """
    def __init__(self, 
                 user_ae:SDAEmodel, 
                 item_ae:SDAEmodel, 
                 field_dims:list,
                 embed_dim:int):
        super().__init__()
        self.user_ae = copy.deepcopy(user_ae)
        self.item_ae = copy.deepcopy(item_ae)

        # Embedding layer
        self.features_embedding = FeaturesEmbedding(field_dims, embed_dim)

        # Get latent representation dimension
        latent_dim = self.user_ae.decoder.mlp[0].in_features and self.item_ae.decoder.mlp[0].in_features 
        # Network
        self.net = MultilayerPerceptrons(input_dim = embed_dim + latent_dim, 
                                         output_layer = "relu")

    def encode_element_wise_product(self, x, user_x, item_x):
        # Obtain latent factor z
        user_z = self.user_ae.encode(user_x)
        item_z = self.item_ae.encode(item_x)

        # Generate embeddings
        x_embed = self.features_embedding(x.int())

        # Seperate user (1st column) and item (2nd column) embeddings from generated embeddings
        user_embed = x_embed[:, 0]
        item_embed = x_embed[:, 1]

        # Concat latent factor and embeddings
        user_repr = torch.cat([user_z, user_embed], dim = 1)
        item_repr = torch.cat([item_z, item_embed], dim = 1)

        return user_repr * item_repr

    def forward(self, x, user_x, item_x):
        # Element wise product between user and items embeddings
        product = self.encode_element_wise_product(x, user_x, item_x)
        
        # Feed element wise product to generalized non-linear layer
        y = self.net(product)

        return y


class GMFPP(BaseModule):
    def __init__(self, 
                 user_ae:SDAEmodel, 
                 item_ae:SDAEmodel, 
                 field_dims:list,
                 embed_dim: int = 8,
                 alpha:Union[int,float] = 1e-3, 
                 beta:Union[int,float] = 1e-3,
                 lr:float = 1e-3,
                 weight_decay:float = 1e-6,
                 criterion:F = F.mse_loss):
        super().__init__(lr, weight_decay, criterion)
        self.model = GMFPPmodel(user_ae, item_ae, field_dims, embed_dim)
        self.alpha = alpha
        self.beta = beta
        self.save_hyperparameters()

    def forward(self, *batch):
        return self.model.forward(*batch)

    def backward_loss(self, *batch):
        x, user_x, item_x, y = batch

        loss = 0
        loss += self.alpha * self.criterion(self.model.user_ae.forward(user_x), user_x)
        loss += self.beta * self.criterion(self.model.item_ae.forward(item_x), item_x)
        loss += super().backward_loss(*batch)

        return loss