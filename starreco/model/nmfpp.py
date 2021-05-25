from typing import Union

import torch
import torch.nn.functional as F

from .module import BaseModule
from .layer import StackedDenoisingAutoEncoder, MultilayerPerceptrons
from .nmf import NMF
from .module import BaseModule
from .gmfpp import GMFPP
from .ncfpp import NCFPP
from .utils import freeze_partial_linear_params

# Done
class NMFPP(BaseModule):
    """
    Neural Matrix Factorization ++ with shared user-item SDAEs.

    - user_sdae_hparams (dict): User SDAE hyperparameteres.
    - item_sdae_hparams (dict): Item SDAE hyperparameteres.
    - nmf_hparams (dict): NeuMF hyperparameteres.
    - nmf_params (dict): NeuMF pretrain weights/parameteers. Default: None.
    - alpha (int/float): Trade-off parameter value for user SDAE. Default: 1e-3.
    - beta (int/float): Trade-off parameter value for item SDAE. Default: 1e-3.
    - lr (float): Learning rate. Default: 1e-3.
    - l2_lambda (float): L2 regularization rate. Default: 1e-3.
    - criterion (F): Criterion or objective or loss function. Default: F.mse_loss.
    """
    
    def __init__(self, 
                 user_sdae_hparams:dict, 
                 item_sdae_hparams:dict, 
                 nmf_hparams:dict,
                 nmf_params:dict = None,
                 alpha:Union[int,float] = 1, 
                 beta:Union[int,float] = 1,
                 lr:float = 1e-3,
                 l2_lambda:float = 0,
                 criterion:F = F.mse_loss):
        assert user_sdae_hparams["hidden_dims"][-1] and item_sdae_hparams["hidden_dims"][-1],\
        "user SDAE and item SDAE latent dimension must be the same"

        super().__init__(lr, 0, criterion)
        self.save_hyperparameters(ignore = ["nmf_params"])
        
        self.l2_lambda = l2_lambda
        self.alpha = alpha
        self.beta = beta

        self.user_sdae = StackedDenoisingAutoEncoder(**user_sdae_hparams)
        self.item_sdae = StackedDenoisingAutoEncoder(**item_sdae_hparams)
        self.nmf = NMF(**nmf_hparams)
        if nmf_params:
            self.nmf.load_state_dict(nmf_params)
            # Obtain first layer input weight before reshaping
            ncf_input_weights = self.nmf.ncf.net.mlp[0].weight.clone()
            nmf_input_weights = self.nmf.net.mlp[0].weight.clone()

        # Replace the first layer with reshape input features
        latent_dim = user_sdae_hparams["hidden_dims"][-1] and item_sdae_hparams["hidden_dims"][-1]
        # NCF layers
        ncf_input_dim = self.nmf.ncf.net.mlp[0].in_features
        ncf_output_dim = self.nmf.ncf.net.mlp[0].out_features
        self.nmf.ncf.net.mlp[0] = torch.nn.Linear(ncf_input_dim + latent_dim * 2, ncf_output_dim)
        # NMF layer
        nmf_input_dim = self.nmf.net.mlp[0].in_features
        nmf_output_dim = self.nmf.net.mlp[0].out_features
        self.nmf.net.mlp[0] = torch.nn.Linear(nmf_input_dim + latent_dim, nmf_output_dim)

        if nmf_params:
            with torch.no_grad():
                # Concat latent weights with pretrained weights
                self.nmf.net.mlp[0].weight[:, -nmf_input_dim:] = nmf_input_weights
                self.nmf.ncf.net.mlp[0].weight[:, :ncf_input_dim] = ncf_input_weights
                if nmf_hparams["freeze_pretrain"]:
                    # Partial freeze input layer
                    freeze_partial_linear_params(self.nmf.ncf.net.mlp[0], list(range(ncf_input_dim)), dim = 1)

    def gmf_element_wise_product(self, x, user_z, item_z):
        # Obtain product of latent factor and embeddings        
        z_product = user_z * item_z
        embed_product = self.nmf.gmf.element_wise_product(x)

        # Concat  embeddings and latent factors
        # Concat order: z 1st then embed *important*
        product = torch.cat([z_product, embed_product], dim = 1) 

        return product
    
    def ncf_forward(self, x, user_z, item_z):
        # Concat latent factor and embeddings        
        z_concat = torch.cat([user_z, item_z], dim = 1)
        embed_concat = self.nmf.ncf.concatenate(x)

        # Concat embeddings and latent factors
        # Concat order: embed 1st then z *important*
        concat = torch.cat([embed_concat, z_concat], dim = 1) 

        # Get output of last hidden layer
        return self.nmf.ncf.net(concat)

    def forward(self, x, user_x, item_x):
        # Obtain latent factor z
        user_z = self.user_sdae.encode(user_x)
        item_z = self.item_sdae.encode(item_x)

        # GMF part: Element wise product between embeddings
        gmf_product = self.gmf_element_wise_product(x, user_z, item_z)

        # NCF part: Concatenate embedding and latent vector and get output of last hidden layer
        ncf_last_hidden = self.ncf_forward(x, user_z, item_z)

        # Concatenate GMF's element wise product and NCF's last hidden layer output
        concat = torch.cat([gmf_product, ncf_last_hidden], dim = 1)

        # Prediction
        y = self.nmf.net(concat)

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
        for param in self.nmf.parameters():
            cf_l2_reg += torch.norm(param).to(self.device)
        rating_loss += self.l2_lambda * cf_l2_reg

        # Total loss
        loss = rating_loss + self.alpha * user_loss + self.beta * item_loss

        return loss        

# Testing
class NMFPPs(BaseModule):
    """
    Neural Matrix Factorization ++ with seperate user-item SDAEs.

    - gmfpp_hparams (dict): GMF++ hyperparameters.
    - ncfpp_hparams (dict): NCF++ hyperparameters.
    - gmfpp_params (dict): GMF++ pretrain weights/parameteers. Default: None.
    - ncfpp_params (dict): NCF++ pretrain weights/parameteers. Default: None.
    - freeze_pretrain (bool): Freeze pretrain weights. Default: True.
    - lr (float): Learning rate. Default: 1e-3.
    - l2_lambda (float): L2 regularization rate. Default: 1e-3.
    - criterion (F): Criterion or objective or loss function. Default: F.mse_loss.
    """

    def __init__(self, 
                 gmfpp_hparams:dict,
                 ncfpp_hparams:dict,
                 gmfpp_params:dict = None,
                 ncfpp_params:dict = None,
                 freeze_pretrain:bool = True,
                 lr:float = 1e-3,
                 l2_lambda:float = 1e-3,
                 criterion:F = F.mse_loss):
        if freeze_pretrain:
            super().__init__(lr, l2_lambda, criterion)
        else:
            super().__init__(lr, 0, criterion)
            self.l2_lambda = l2_lambda
        self.save_hyperparameters(ignore = ["gmfpp_params", "ncfpp_params"])

        self.gmfpp = GMFPP(**gmfpp_hparams)
        self.ncfpp = NCFPP(**ncfpp_hparams)

        # Load pretrained weights
        if gmfpp_params:
            self.gmfpp.load_state_dict(gmfpp_params)
        if ncfpp_params:
            self.ncfpp.load_state_dict(ncfpp_params)

        # Freeze model
        if freeze_pretrain:
            self.gmfpp.freeze()
            self.ncfpp.freeze()

        # Remove GMF output layer
        del self.gmfpp.gmf.net

        # Remove NCF output layer
        del self.ncfpp.ncf.net.mlp[-1]
        if type(self.ncfpp.ncf.net.mlp[-1]) == torch.nn.Linear:
            del self.ncfpp.ncf.net.mlp[-1]

        # Add input dim
        input_dim = gmfpp_hparams["gmf_hparams"]["embed_dim"]
        input_dim += gmfpp_hparams["user_sdae_hparams"]["hidden_dims"][-1] and gmfpp_hparams["item_sdae_hparams"]["hidden_dims"][-1]
        input_dim += ncfpp_hparams["ncf_hparams"]["hidden_dims"][-1]
        self.net = MultilayerPerceptrons(input_dim = input_dim, 
                                         output_layer = "relu")
        self.freeze_pretrain = freeze_pretrain

    def forward(self, x, user, item):
        # GMFPP part: Element wise product between latent factor and embeddings
        gmfpp_product = self.gmfpp.encode_element_wise_product(x, user, item)

        # NCFPP part: Get output of last hidden layer
        ncfpp_last_hidden = self.ncfpp.forward(x, user ,item)

        # Concatenate GMFPP's element wise product and NCFPP's last hidden layer output
        concat = torch.cat([gmfpp_product, ncfpp_last_hidden], dim = 1)

        # Prediction
        y = self.net(concat)

        return y

    def backward_loss(self, *batch):
        """Custom backward loss"""
        rating_loss = super().backward_loss(*batch)

        if self.freeze_pretrain:
            # If freeze pretrain, only take accunt the rating loss for back propagration
            return rating_loss
        else:
            # GMF++ loss which include GMF++ rating loss, reconstruction loss
            gmfpp_loss = self.gmfpp.backward_loss(*batch)
            # NCFP++ loss which include NCF++ rating loss and reconstruction loss
            ncfpp_loss = self.ncfpp.backward_loss(*batch)
            # Rating loss
            loss = rating_loss + gmfpp_loss + ncfpp_loss

            return loss
