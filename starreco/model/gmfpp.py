from typing import Union

import torch
import torch.nn.functional as F

from starreco.model import (FeaturesEmbedding, 
                            MultilayerPerceptrons,
                            Module)
from .ae import AE
from .dae import DAE

class GMFpp(Module):
    """
    GMF++ or Generalized Matrix Factorization plus plus
    """
    def __init__(self, user_ae:Union[AE, DAE], item_ae:Union[AE, DAE], 
                 activation:str = "relu",
                 dropout = 0.5,
                 lr:float = 1e-2,
                 weight_decay:float = 0,
                 criterion:F = F.mse_loss):
        """
        Hyperparameters setting.

        :param feature_dims (list): List of feature dimensions. 

        :param embed_dim (int): Embeddings dimensions. Default: 8

        :param activation (str): Activation Function. Default: "relu".

        :param lr (float): Learning rate. Default: 1e-3

        :param weight_decay (float): L2 regularization weight decay: Default: 1e-3

        :param criterion (F): Objective function. Default: F.mse_loss
        """
        super().__init__(lr, weight_decay, criterion)

        self.user_ae = user_ae
        for i, module in enumerate(self.user_ae.decoder.mlp):
            if type(module) == torch.nn.Linear:
                try:
                    latent_dim
                except:
                    latent_dim = self.user_ae.decoder.mlp[i].weight.shape[1]
                else:
                    pass
                self.user_ae.decoder.mlp[i].weight.requires_grad = False
                self.user_ae.decoder.mlp[i].bias.requires_grad = False
                self.user_dim = self.user_ae.decoder.mlp[i].weight.shape[0]

        self.item_ae = item_ae
        for i, module in enumerate(self.item_ae.decoder.mlp):
            if type(module) == torch.nn.Linear:
                self.item_ae.decoder.mlp[i].weight.requires_grad = False
                self.item_ae.decoder.mlp[i].bias.requires_grad = False
                self.item_dim = self.item_ae.decoder.mlp[i].weight.shape[0]

        # Multilayer perceptrons
        self.mlp = MultilayerPerceptrons(input_dim = latent_dim, 
                                         activations = activation, 
                                         dropouts = dropout,
                                         output_layer = "relu")

    def forward(self, x):
        """
        Perform operations.

        :x (torch.tensor): Input tensors of shape (batch_size, user_ae.io_dim + item_ae.io_dim).

        :return (torch.tensor): Output prediction tensors of shape (batch_size, 1)
        """
        user_latent = self.user_ae.encode(x[:, :self.user_dim])
        item_latent = self.item_ae.encode(x[:, -self.item_dim:])

        # Element wise product between embeddings
        product = user_latent * item_latent

        # Prediction
        return self.mlp(product)
