from typing import Union

import torch
import torch.nn.functional as F

from .gmf import GMF
from .layers import StackedDenoisingAutoEncoder

# Testing
class GMFPP(GMF):
    """
    Generalized Matrix Factorization ++.

    - user_sdae_kwargs (dict): User SDAE hyperparameteres.
    - item_sdae_kwargs (dict): Item SDAE hyperparameteres.
    - field_dims (list): List of features dimensions.
    - embed_dim (int): Embedding dimension. Default: 8.
    - alpha (int/float): Trade off parameter for user feature reconstruction. Default: 1.
    - beta (int/float): Trade off parameter for item feature reconstruction. Default: 1.
    - lr (float): Learning rate. Default: 1e-3.
    - weight_decay (float): L2 regularization rate. Default: 1e-3.
    - criterion: Criterion or objective or loss function. Default: F.mse_loss.
    """

    def __init__(self, 
                 user_sdae_kwargs:dict, 
                 item_sdae_kwargs:dict, 
                 field_dims:list, 
                 embed_dim:int = 8, 
                 alpha:Union[int,float] = 1, 
                 beta:Union[int,float] = 1,
                 lr:float = 1e-3,
                 weight_decay:float = 1e-6,
                 criterion = F.mse_loss):
        assert user_sdae_kwargs["hidden_dims"][-1] and item_sdae_kwargs["hidden_dims"][-1],\
        "`user_sdae_kwargs` and `item_sdae_kwargs` last `hidden_dims` (latent dimension) must be the same"

        super().__init__(field_dims, embed_dim, lr, weight_decay, criterion)
        self.save_hyperparameters()

        self.alpha = alpha
        self.beta = beta

        # Stacked denoising autoencoder for feature extraction
        self.user_sdae = StackedDenoisingAutoEncoder(**user_sdae_kwargs)
        self.item_sdae = StackedDenoisingAutoEncoder(**item_sdae_kwargs)

        # Replace the first layer with reshape input features
        latent_dim = user_sdae_kwargs["hidden_dims"][-1] and item_sdae_kwargs["hidden_dims"][-1]
        input_dim = self.net.mlp[0].in_features
        output_dim = self.net.mlp[0].out_features
        self.net.mlp[0] = torch.nn.Linear(input_dim + latent_dim, output_dim)        

    def forward(self, x, user_x, item_x):
        # Element wise product between user and item embeddings
        x_embed = self.features_embedding(x.int())
        user_embed, item_embed = x_embed[:, 0], x_embed[:, 1]
        embed_product = user_embed * item_embed

        # Element wise product between user and item latent representations
        user_z, item_z = self.user_sdae.encode(user_x), self.item_sdae.encode(item_x)     
        z_product = user_z * item_z

        # Concatenate user and item embeddings and latent representations
        concat = torch.cat([embed_product, z_product], dim = 1)
        
        # Non linear on concatenated vectors
        y = self.net(concat)

        return y        

    def backward_loss(self, *batch):
        """
        Custom backward loss.
        """
        x, user_x, item_x, y = batch
        # User reconstruction loss with trade off parameter alpha
        user_loss = self.alpha * self.criterion(self.user_sdae.forward(user_x), user_x)
        # Item reconstruction loss with trade off parameter beta
        item_loss = self.beta * self.criterion(self.item_sdae.forward(item_x), item_x)
        # Rating loss
        rating_loss = super().backward_loss(*batch)

        # Total loss
        loss = rating_loss + user_loss +  item_loss

        return loss

    def load_pretrain_weights(self, gmf_weights):
        """
        Load pretrain GMF weights.
        """
        gmfpp_weights = self.state_dict()                 

        gmf_input_weights = gmf_weights["net.mlp.0.weight"]
        gmf_input_dim = gmf_input_weights.shape[-1]
        gmf_weights["net.mlp.0.weight"] = gmfpp_weights["net.mlp.0.weight"]
        gmf_weights["net.mlp.0.weight"][:, :gmf_input_dim] = gmf_input_weights
        gmfpp_weights.update(gmf_weights)               
        self.load_state_dict(gmfpp_weights)