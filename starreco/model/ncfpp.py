from typing import Union
import copy

import torch
import torch.nn.functional as F

from .module import BaseModule
from .layer import FeaturesEmbedding, MultilayerPerceptrons
from .sdae import SDAE, SDAEModule

class NCFPP(torch.nn.Module):
    """
    Neural Collaborative Filtering/Multilayer Perceptrons ++
    """
    def __init__(self, user_ae:SDAE, item_ae:SDAE, field_dims:list, 
                 embed_dim:int = 16,
                 hidden_dims:list = [32, 16, 8], 
                 activations:Union[str, list] = "relu", 
                 dropouts:Union[float, list] = 0.5, 
                 batch_norm:bool = True):
        super().__init__()
        self.user_ae = copy.deepcopy(user_ae)
        self.item_ae = copy.deepcopy(item_ae)

        self.user_feature_dim = self.user_ae.decoder.mlp[0].weight.shape[0]
        self.item_feature_dim = self.item_ae.decoder.mlp[0].weight.shape[0]

        # Obtain latent factor dimension
        for i, module in enumerate(self.user_ae.decoder.mlp):
            if type(module) == torch.nn.Linear:
                try: latent_dim
                except: latent_dim = self.user_ae.decoder.mlp[i].weight.shape[1]
                else: pass

        # Embedding layer
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)

        # Multilayer perceptrons
        self.mlp = MultilayerPerceptrons(input_dim = embed_dim * 2 + latent_dim * 2, 
                                         hidden_dims = hidden_dims, 
                                         activations = activations, 
                                         dropouts = dropouts,
                                         output_layer = "relu",
                                         batch_norm = batch_norm)

    def forward(self, x):
        # Partition user and item features
        user_feature = x[:, 2:self.user_feature_dim + 2]
        item_feature = x[:, -self.item_feature_dim:]
        x = x[:, :2].int()

         # Obtain latent factor
        user_latent = self.user_ae.encode(user_feature)
        item_latent = self.item_ae.encode(item_feature)

        # Generate embeddings
        embed = self.embedding(x)
        # Seperate user (1st column) and item (2nd column) embeddings from generated embeddings
        user_embed = embed[:, 0]
        item_embed = embed[:, 1]

        # Concat latent factor and embeddings
        user_latent = torch.cat([user_latent, user_embed], dim = 1)
        item_latent = torch.cat([item_latent, item_embed], dim = 1)

        # Concat user and item latent factors
        concat = torch.cat([user_latent, item_latent], dim = 1)

        # Feed element wise product to generalized non-linear layer
        y = self.mlp(concat)

        return y

class NCFPPModule(BaseModule):
    def __init__(self, user_ae, item_ae, field_dims:list, 
                 embed_dim:int = 16,
                 hidden_dims:list = [32, 16, 8], 
                 activations:Union[str, list] = "relu", 
                 dropouts:Union[float, list] = 0.5, 
                 batch_norm:bool = True,
                 lr:float = 1e-3,
                 weight_decay:float = 1e-6,
                 criterion:F = F.mse_loss,
                 save_hyperparameters:bool = True):
        super().__init__(lr, weight_decay, criterion)
        self.model = NCFPP(user_ae, item_ae, field_dims, embed_dim, hidden_dims, activations, dropouts, batch_norm)
        self.save_hyperparameters()

    def evaluate(self, x, y): 
        # Partition user and item features
        user_feature = x[:, 2:self.model.user_feature_dim + 2]
        item_feature = x[:, -self.model.item_feature_dim:]
        user_loss = self.criterion(self.model.user_ae.forward(user_feature), user_feature)
        item_loss = self.criterion(self.model.item_ae.forward(item_feature), item_feature)
        ncf_loss = super().evaluate(x, y)

        loss = user_loss + item_loss + ncf_loss 

        return loss