from typing import Union

import torch
import torch.nn.functional as F

from .module import BaseModule
from .layer import StackedDenoisingAutoEncoder
from .gmf import GMF

# Done
class GMFPP(BaseModule):
    """
    Generalized Matrix Factorization ++.

    - user_sdae_hparams (dict): User SDAE hyperparameteres.
    - item_sdae_hparams (dict): Item SDAE hyperparameteres.
    - gmf_hparams (dict): GMF hyperparameteres.
    - gmf_params (dict): GMF pretrain weights/parameteers. Default: None.
    - alpha (int/float): Trade-off parameter value for user SDAE. Default: 1e-3.
    - beta (int/float): Trade-off parameter value for item SDAE. Default: 1e-3.
    - lr (float): Learning rate. Default: 1e-3.
    - l2_lambda (float): L2 regularization rate. Default: 1e-3.
    - criterion (F): Criterion or objective or loss function. Default: F.mse_loss.
    """

    def __init__(self, 
                 user_sdae_hparams:dict, 
                 item_sdae_hparams:dict, 
                 gmf_hparams:dict,
                 gmf_params:dict = None,
                 alpha:Union[int,float] = 1, 
                 beta:Union[int,float] = 1,
                 lr:float = 1e-3,
                 l2_lambda:float = 1e-3,
                 criterion:F = F.mse_loss):
        assert user_sdae_hparams["hidden_dims"][-1] and item_sdae_hparams["hidden_dims"][-1],\
        "user SDAE and item SDAE latent dimension must be the same"

        super().__init__(lr, 0, criterion)
        self.save_hyperparameters(ignore = ["gmf_params"])
        
        self.l2_lambda = l2_lambda
        self.alpha = alpha
        self.beta = beta

        self.user_sdae = StackedDenoisingAutoEncoder(**user_sdae_hparams)
        self.item_sdae = StackedDenoisingAutoEncoder(**item_sdae_hparams)
        self.gmf = GMF(**gmf_hparams)
        if gmf_params: self.gmf.load_state_dict(gmf_params)

        # Replace the first layer with reshape input features
        latent_dim = user_sdae_hparams["hidden_dims"][-1] and item_sdae_hparams["hidden_dims"][-1]
        input_dim = self.gmf.net.mlp[0].in_features
        output_dim = self.gmf.net.mlp[0].out_features
        # Obtain first layer weights before reshaping
        input_weights = self.gmf.net.mlp[0].weight.clone()
        self.gmf.net.mlp[0] = torch.nn.Linear(input_dim + latent_dim, output_dim)

        if gmf_params:
            with torch.no_grad():
                # Concat latent weights with pretrained weights
                self.gmf.net.mlp[0].weight[:, :input_dim] = input_weights

    def encode_element_wise_product(self, x, user_x, item_x):
        # Obtain latent factor z
        user_z = self.user_sdae.encode(user_x)
        item_z = self.item_sdae.encode(item_x)

        # Obtain product of latent factor and embeddings        
        z_product = user_z * item_z
        embed_product = self.gmf.element_wise_product(x)

        # Concat  embeddings and latent factors
        product = torch.cat([embed_product, z_product], dim = 1)

        return product

    def forward(self, x, user_x, item_x):
        # Element wise product between user and items embeddings
        product = self.encode_element_wise_product(x, user_x, item_x)

        # Feed element wise product to generalized non-linear layer
        y = self.gmf.net(product)

        return y

    def backward_loss(self, *batch):
        """Custom backward loss"""
        x, user_x, item_x, y = batch

        # User reconstruction loss with trade off parameter alpha
        user_loss = self.criterion(self.user_sdae.forward(user_x), user_x)
        user_l2_reg = torch.tensor(0.).to(self.device)
        for param in self.user_sdae.parameters():
            user_l2_reg += torch.norm(param).to(self.device)
        user_loss += self.l2_lambda * user_l2_reg

        # Item reconstruction loss with trade off parameter beta
        item_loss = self.criterion(self.item_sdae.forward(item_x), item_x)
        item_l2_reg = torch.tensor(0.).to(self.device)
        for param in self.item_sdae.parameters():
            item_l2_reg += torch.norm(param).to(self.device)
        item_loss += self.l2_lambda * item_l2_reg

        # Rating loss
        rating_loss = super().backward_loss(*batch)
        cf_l2_reg = torch.tensor(0.).to(self.device)
        for param in self.gmf.parameters():
            cf_l2_reg += torch.norm(param).to(self.device)
        rating_loss += self.l2_lambda * cf_l2_reg

        # Total loss
        loss = rating_loss + self.alpha * user_loss + self.beta * item_loss

        return loss