from typing import Union

import torch
import torch.nn.functional as F

from starreco.model import (FeaturesEmbedding, 
                            MultilayerPerceptrons,
                            Module)
from .dae import DAE

class MLPpp(Module):
    """
    Neural Collaborative Filtering Framework's Multilayer Perceptrons ++
    """
    def __init__(self, user_ae_params:dict, item_ae_params:dict, field_dims:list, 
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

       # Stacked Denoising Autoencoder 
        # User SDAE for user feature reconstruction
        self.user_ae = DAE(**user_ae_params)
        self.user_ae.save_hyperparameters = False
        self.user_feature_dim = self.user_ae.decoder.mlp[0].weight.shape[0]
        # Item SFAE for item feature reconstruction
        self.item_ae = DAE(**item_ae_params)
        self.item_ae.save_hyperparameters = False
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
        self.nn = MultilayerPerceptrons(input_dim = embed_dim * 2 + latent_dim * 2, 
                                        hidden_dims = hidden_dims, 
                                        activations = activations, 
                                        dropouts = dropouts,
                                        output_layer = "relu",
                                        batch_norm = batch_norm)
        
        # Save hyperparameters to checkpoint
        if save_hyperparameters:
            self.save_hyperparameters()

    def forward(self, x):
        """
        Perform operations.

        :x (torch.tensor): Input tensors of shape (batch_size, 2) user and item.

        :return (torch.tensor): Output prediction tensors of shape (batch_size, 1)
        """
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
        y = self.nn(concat)

        return y

    def evaluate(self, x, y): 
        # Partition user and item features
        user_feature = x[:, 2:self.user_feature_dim + 2]
        item_feature = x[:, -self.item_feature_dim:]

        user_loss = self.user_ae.evaluate(user_feature, user_feature)
        item_loss = self.item_ae.evaluate(item_feature, item_feature)
        mlp_loss = super().evaluate(x, y)

        loss = user_loss + item_loss + mlp_loss 

        return loss
