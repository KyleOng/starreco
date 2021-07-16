from typing import Union

import torch
import torch.nn.functional as F

from .gmf import GMF
from .layers import StackedDenoisingAutoEncoder
from .utils import l2_regularization

# Testing
class GMFPP(GMF):
    """
    Generalized Matrix Factorization ++.

    - user_sdae_hparams (dict): User SDAE hyperparameteres.
    - item_sdae_hparams (dict): Item SDAE hyperparameteres.
    - field_dims (list): List of features dimensions.
    - embed_dim (int): Embedding dimension. Default: 8.
    - alpha (int/float): Trade off parameter for user feature reconstruction. Default: 1.
    - beta (int/float): Trade off parameter for item feature reconstruction. Default: 1.
    - lr (float): Learning rate. Default: 1e-3.
    - weight_decay (float): L2 regularization rate. Default: 1e-3.
    - criterion: Criterion or objective or loss function. Default: F.mse_loss.
    """

    def __init__(self, 
                 user_sdae_hparams:dict, 
                 item_sdae_hparams:dict, 
                 field_dims:list, 
                 embed_dim:int = 8, 
                 alpha:Union[int,float] = 1, 
                 beta:Union[int,float] = 1,
                 lr:float = 1e-3,
                 weight_decay:float = 1e-6,
                 user_weight_decay:float = 1e-6,
                 item_weight_decay:float = 1e-6,
                 criterion = F.mse_loss,
                 user_criterion = F.mse_loss,
                 item_criterion = F.mse_loss):
        assert user_sdae_hparams["hidden_dims"][-1] and item_sdae_hparams["hidden_dims"][-1],\
        "`user_sdae_hparams` and `item_sdae_hparams` last `hidden_dims` (latent dimension) must be the same"

        # Set weight decay as 0 in super()__.init__ as we manually set the regularizer
        super().__init__(field_dims, embed_dim, lr, 0, criterion)
        self.save_hyperparameters()

        self.alpha = alpha
        self.beta = beta
        self.weight_decay = weight_decay
        self.user_weight_decay = user_weight_decay
        self.item_weight_decay = item_weight_decay
        self.user_criterion = user_criterion
        self.item_criterion = item_criterion

        # Stacked denoising autoencoder for feature extraction
        self.user_sdae = StackedDenoisingAutoEncoder(**user_sdae_hparams)
        self.item_sdae = StackedDenoisingAutoEncoder(**item_sdae_hparams)

        # Replace the first layer with reshape input features
        latent_dim = user_sdae_hparams["hidden_dims"][-1] and item_sdae_hparams["hidden_dims"][-1]
        input_dim = self.net.mlp[0].in_features
        output_dim = self.net.mlp[0].out_features
        self.net.mlp[0] = torch.nn.Linear(input_dim + latent_dim, output_dim)        

    def forward(self, x, user_x, item_x):
        # Element wise product between user and item embeddings
        x_embed = self.features_embedding(x.int())
        user_embed, item_embed = x_embed[:, 0], x_embed[:, 1]
        embed_product = user_embed * item_embed

        # Element wise product between user and item latent features
        # Perform user and features reconstruction (add noise during reconstruction)
        user_y, item_y = self.user_sdae.forward(user_x), self.item_sdae.forward(item_x)   
        # Extract user and item latent features (remove noise during extraction)
        user_z, item_z = self.user_sdae.encode(user_x, add_noise = False), self.item_sdae.encode(item_x, add_noise = False) 
        z_product = user_z * item_z

        # Concatenate user and item embeddings and latent features
        concat = torch.cat([embed_product, z_product], dim = 1)
        
        # Non linear on concatenated vectors
        y = self.net(concat)

        return y, user_y, item_y  

    def reconstruction_loss(self, user_x, item_x, user_x_hat, item_x_hat):
        """
        Reconstruction loss.
        """
        # User reconstruction loss
        user_loss = self.user_criterion(user_x_hat, user_x)
        user_reg = l2_regularization(self.user_weight_decay, self.user_sdae.parameters(), self.device)
        user_loss *= self.alpha
        user_loss += user_reg

        # Item reconstruction loss
        item_loss = self.item_criterion(item_x_hat, item_x)
        item_reg = l2_regularization(self.item_weight_decay, self.item_sdae.parameters(), self.device)
        item_loss *= self.alpha
        item_loss += item_reg

        return user_loss + item_loss

    def backward_loss(self, *batch):
        """
        Custom backward loss.
        """
        x, user_x, item_x, y = batch

        # Prediction
        y_hat, user_x_hat, item_x_hat = self.forward(x, user_x, item_x)

        # Reconstruction loss
        reconstruction_loss = self.reconstruction_loss(user_x, item_x, user_x_hat, item_x_hat)

        # Rating loss
        rating_loss = self.criterion(y_hat, y)
        rating_reg = l2_regularization(self.weight_decay, super().parameters(), self.device)
        rating_loss += rating_reg

        # Total loss
        loss = rating_loss + reconstruction_loss

        return loss

    def logger_loss(self, *batch):
        """
        Overwrite logger loss which focus evaluation on y_hat only
        """
        xs = batch[:-1]
        y = batch[-1]
        preds = self.forward(*xs)
        loss = self.criterion(preds[0], y)

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