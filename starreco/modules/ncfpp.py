from typing import Union

import torch
import torch.nn.functional as F

from .ncf import NCF
from .layers import StackedDenoisingAutoEncoder

# Done
class NCFPP(NCF):
    """
    Neural Collaborative Filtering ++.

    - user_sdae_kwargs (dict): User SDAE hyperparameteres.
    - item_sdae_kwargs (dict): Item SDAE hyperparameteres.
    - field_dims (list): List of features dimensions.
    - embed_dim (int): Embedding dimension.
    - hidden_dims (list): List of numbers of neurons throughout the hidden layers. Default: [32,16,8].
    - activations (str/list): List of activation functions. Default: "relu".
    - dropouts (int/float/list): List of dropout values. Default: 0.5.
    - batch_norm (bool): If True, apply batch normalization in every layer. Batch normalization is applied between activation and dropout layer. Default: True.
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
                 hidden_dims:list = [32, 16, 8], 
                 activations:Union[str, list] = "relu", 
                 dropouts:Union[int, float, list] = 0.5, 
                 batch_norm:bool = True,
                 alpha:Union[int,float] = 1, 
                 beta:Union[int,float] = 1,
                 lr:float = 1e-3,
                 weight_decay:float = 1e-3,
                 criterion = F.mse_loss):
        assert user_sdae_kwargs["hidden_dims"][-1] and item_sdae_kwargs["hidden_dims"][-1],\
        "`user_sdae_kwargs` and `item_sdae_kwargs` last `hidden_dims` (latent dimension) must be the same"

        super().__init__(field_dims, embed_dim, hidden_dims, activations, dropouts, batch_norm, lr, weight_decay, criterion)
        self.save_hyperparameters()
                
        self.alpha = alpha
        self.beta = beta

        # Stacked denoising autoencoder for feature extraction.
        self.user_sdae = StackedDenoisingAutoEncoder(**user_sdae_kwargs)
        self.item_sdae = StackedDenoisingAutoEncoder(**item_sdae_kwargs)

        # Replace the first layer with reshape input features
        latent_dim = user_sdae_kwargs["hidden_dims"][-1] and item_sdae_kwargs["hidden_dims"][-1]
        input_dim = self.net.mlp[0].in_features
        output_dim = self.net.mlp[0].out_features
        self.net.mlp[0] = torch.nn.Linear(input_dim + latent_dim * 2, output_dim)       

    def forward(self, x, user_x, item_x):
        # Concatenate user and item embeddings
        x_embed = self.features_embedding(x.int())
        embed_concat = torch.flatten(x_embed, start_dim = 1)

        # Concatenate user and item latent representations.
        user_z, item_z = self.user_sdae.encode(user_x), self.item_sdae.encode(item_x)     
        z_concat = torch.cat([user_z, item_z], dim = 1)

        # Concatenate user and item emebddings and latent representations
        concat = torch.cat([embed_concat, z_concat], dim = 1)
        
        # Non linear on concatenated vectors
        y = self.net(concat)

        return y     

    def backward_loss(self, *batch):
        """
        Custom backward loss
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

    def load_pretrain_weights(self, ncf_weights):
        """
        Load pretrain NCF weights
        """
        ncfpp_weights = self.state_dict()                 

        ncf_input_weights = ncf_weights["net.mlp.0.weight"]
        ncf_input_dim = ncf_input_weights.shape[-1]
        ncf_weights["net.mlp.0.weight"] = ncfpp_weights["net.mlp.0.weight"]
        ncf_weights["net.mlp.0.weight"][:, :ncf_input_dim] = ncf_input_weights
        ncfpp_weights.update(ncf_weights)               
        self.load_state_dict(ncfpp_weights)