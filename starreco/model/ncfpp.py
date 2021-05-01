from typing import Union
import copy

import torch
import torch.nn.functional as F

from .module import BaseModule
from .layer import FeaturesEmbedding, MultilayerPerceptrons
from .sdae import SDAEmodel

class NCFPPmodel(torch.nn.Module):
    """
    Neural Collaborative Filtering Multilayer Perceptrons ++ model
    """
    def __init__(self, 
                 user_ae:SDAEmodel, 
                 item_ae:SDAEmodel, 
                 field_dims:list, 
                 embed_dim:int,
                 hidden_dims:list, 
                 activations:Union[str, list], 
                 dropouts:Union[int, float, list], 
                 batch_norm:bool):
        super().__init__()
        self.user_ae = copy.deepcopy(user_ae)
        self.item_ae = copy.deepcopy(item_ae)

        # Embedding layer
        self.features_embedding = FeaturesEmbedding(field_dims, embed_dim)

        # Get latent representation dimension
        latent_dim = self.user_ae.decoder.mlp[0].in_features and self.item_ae.decoder.mlp[0].in_features 
        # Network 
        self.net = MultilayerPerceptrons(input_dim = embed_dim * 2 + latent_dim * 2, 
                                         hidden_dims = hidden_dims, 
                                         activations = activations, 
                                         dropouts = dropouts,
                                         output_layer = "relu",
                                         batch_norm = batch_norm)

    def encode_concatenate(self, x, user_x, item_x):
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

        return torch.cat([user_repr, item_repr], dim = 1)

    def forward(self, x, user_x, item_x):
        # Element wise product between user and items embeddings
        concat = self.encode_concatenate(x, user_x, item_x)
        
        # Feed element wise product to generalized non-linear layer
        y = self.net(concat)

        return y


class NCFPP(BaseModule):
    def __init__(self, 
                 user_ae:SDAEmodel, 
                 item_ae:SDAEmodel, 
                 field_dims:list, 
                 embed_dim:int = 8,
                 hidden_dims:list = [32, 16, 8], 
                 activations:Union[str, list] = "relu", 
                 dropouts:Union[int, float, list] = 0.5, 
                 batch_norm:bool = True,
                 alpha:Union[int,float] = 0.1, 
                 beta:Union[int,float] = 0.1,
                 lr:float = 1e-3,
                 weight_decay:float = 1e-3,
                 criterion:F = F.mse_loss,
                 save_hyperparameters:bool = True):
        super().__init__(lr, weight_decay, criterion)
        self.model = NCFPPmodel(user_ae, item_ae, field_dims, embed_dim, hidden_dims, activations, dropouts, batch_norm)
        self.alpha = alpha
        self.beta = beta
        self.save_hyperparameters()

    def forward(self, *batch):
        return self.model.forward(*batch)

    def backward_loss(self, *batch):
        x, user_x, item_x, y = batch

        loss = 0
        loss += self.alpha * self.criterion(self.model.user_ae.forward(user_x), user_x)
        loss += self.beta  * self.criterion(self.model.item_ae.forward(item_x), item_x)
        loss += super().backward_loss(*batch)

        return loss