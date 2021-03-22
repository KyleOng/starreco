from typing import Union
import copy

import torch
import torch.nn.functional as F

from starreco.model import (MultilayerPerceptrons, 
                            Module)

class VAE(Module):
    """
    Variational Autoencoder
    """
    def __init__(self, io_dim:int,
                 hidden_dims:list = [512, 256, 64], 
                 e_activations:Union[str, list] = "relu", 
                 d_activations:Union[str, list] = "relu", 
                 dropout:float = 0.5,
                 dense_refeeding:int = 1,
                 batch_norm:bool = True,
                 lr:float = 1e-3,
                 weight_decay:float = 1e-3,
                 criterion:F = F.mse_loss):
        """
        Hyperparameters setting.

        :param io_dim (int): Input/Output dimension.

        :param hidden_dims (list): List of number of hidden nodes for encoder and decoder (in reverse). For example, hidden_dims [200, 100, 50] = encoder [io_dim, 200, 100, 50], decoder [50, 100, 200, io_dim]. Default: [512, 256, 128]

        :param e_activations (str/list): List of activation functions for encoder layer. If type str, then the activation will be repeated len(hidden_dims) times in a list. Default: "relu"

        :param d_activations (str/list): List of activation functions for decoder layer. If type str, then the activation will be repeated len(hidden_dims) times in a list. Default: "relu"

        :param dropout (float): Dropout within latent space. Default: 0.5

        :param dense_refeeding (int): Number of dense refeeding: Default: 1

        :param batch_norm (bool): If True, apply batch normalization on every hidden layer. Default: True

        :param lr (float): Learning rate. Default: 1e-3

        :param weight_decay (float): L2 regularization weight decap: Default: 1e-3

        :param criterion (F): Objective function. Default: F.mse_loss
        """
        super().__init__(lr, weight_decay, criterion)
        self.dense_refeeding = dense_refeeding

        # Encoder layer
        self.encoder = MultilayerPerceptrons(input_dim = io_dim,
                                             hidden_dims = hidden_dims[:-1], 
                                             activations = e_activations, 
                                             dropouts = 0,
                                             output_layer = None,
                                             batch_norm = batch_norm)
        self.mean = MultilayerPerceptrons(input_dim = hidden_dims[-2],
                                          hidden_dims = [hidden_dims[-1]],
                                          activations = e_activations, 
                                          dropouts = 0,
                                          output_layer = None,
                                          batch_norm = batch_norm)
        self.std = copy.deepcopy(self.mean)

        # Dropout layer in latent space
        self.dropout = torch.nn.Dropout(dropout)

        # Decoder layer 
        self.decoder = MultilayerPerceptrons(input_dim = hidden_dims[-1],
                                             hidden_dims = [*hidden_dims[:-1][::-1], io_dim], 
                                             activations = d_activations, 
                                             dropouts = 0,
                                             apply_last_hidden = False,
                                             output_layer = None,
                                             batch_norm = batch_norm)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x):
        x = self.encoder(x)
        mu, logvar = self.mean(x), self.std(x)
        x = self.reparameterize(mu, logvar) 
        return x, mu, logvar

    def forward(self, x):
        for i in range(self.dense_refeeding):
            x, _, _ = self.encode(x)
            x = self.decoder(self.dropout(x))

        return x
