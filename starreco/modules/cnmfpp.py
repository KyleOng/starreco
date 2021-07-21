from typing import Union

import torch
import torch.nn.functional as F

from .nmfpp import NMFPP
from .layers import DocumentContextAnalysis, MultilayerPerceptrons
from .utils import l2_regularization

class CNMFPP(NMFPP):
    """
    Convolutional Neural Matrix Factorization ++.
    """

    def __init__(self, 
                 gmfpp_hparams:dict,
                 ncfpp_hparams:dict,
                 word_dim:int,
                 max_len:int, 
                 shared_embed:str= None,
                 shared_sdaes:str= None,
                 output_dim:int = 50,
                 word_embed_dim:int = 200,
                 filter_size:list = 100,
                 kernel_size:int = 3,
                 c_activation:str = "relu",
                 c_dropout:Union[int, float] = 0,
                 fc_dim:int = 200,
                 fc_activation:str = "tanh", 
                 fc_dropout:Union[int, float] = 0.2,
                 batch_norm:bool = True,
                 lr:float = 1e-5,
                 weight_decay:float = 1e-3,
                 criterion = F.mse_loss):
        super().__init__(gmfpp_hparams, ncfpp_hparams, shared_embed, shared_sdaes, lr, weight_decay, criterion)
        self.max_len = max_len
        self.cnn = DocumentContextAnalysis(word_dim, max_len, output_dim, word_embed_dim, filter_size, kernel_size, c_activation, c_dropout, fc_dim, fc_activation, fc_dropout, batch_norm)

        # Reshape NeuMF (prediction) layer input features shape
        if fc_dim:
            input_dim = self.net.mlp[0].in_features + output_dim
        else:
            input_dim = self.net.mlp[0].in_features + filter_size
        self.net = MultilayerPerceptrons(input_dim = input_dim,  output_layer = "relu")

    def forward(self, x, user_x, item_x):
        # Seperate document from item
        document_x = item_x[:, -self.max_len:]
        item_x = item_x[:, :-self.max_len]

        # Fusion of GMF++ and NCF++
        outputs = self.fusion(x, user_x, item_x)

        # Get latent document vector
        latent_document = self.cnn(document_x)

        # Fusion of GMF++, NCF++ and CNN
        concat = torch.cat([outputs[0], latent_document], dim = 1)

        # non-linear on features
        outputs[0] = self.net(concat)

        return outputs

    def l2_regularization(self):
        """
        Add CNN regularization to NMF++ regularization.
        """
        cnn_reg = l2_regularization(self.weight_decay, self.cnn.parameters(), self.device)
        return super().l2_regularization() + cnn_reg

    def backward_loss(self, *batch):
        """
        Custom backward loss.
        """

        x, user_x, item_x, y = batch

        if self.shared_sdaes== "gmf++":
            # Prediction
            y_hat, user_x_hat, item_x_hat = self.forward(x, user_x, item_x)
            # Reconstruction loss
            reconstruction_loss = self.gmfpp.reconstruction_loss(user_x, item_x[:, :-self.max_len], user_x_hat, item_x_hat)
        elif self.shared_sdaes == "ncf++":
            # Prediction
            y_hat, user_x_hat, item_x_hat = self.forward(x, user_x, item_x)
            # Reconstruction loss
            reconstruction_loss = self.ncfpp.reconstruction_loss(user_x, item_x[:, :-self.max_len], user_x_hat, item_x_hat)
        else:
            # Prediction
            y_hat, user_x_hat_gmfpp, item_x_hat_gmfpp, user_x_hat_ncfpp, item_x_hat_ncfpp = self.forward(x, user_x, item_x)
            # Reconstruction loss
            reconstruction_loss = self.gmfpp.reconstruction_loss(user_x, item_x[:, :-self.max_len], user_x_hat_gmfpp, item_x_hat_gmfpp)
            reconstruction_loss += self.ncfpp.reconstruction_loss(user_x, item_x[:, :-self.max_len], user_x_hat_ncfpp, item_x_hat_ncfpp)

        # Rating loss
        rating_loss = self.criterion(y_hat, y)
        
        # L2 regularization
        reg = self.l2_regularization()

        # Total loss
        loss = rating_loss + reconstruction_loss + reg
        
        return loss