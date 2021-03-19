from typing import Union

import torch
import torch.nn.functional as F

from starreco.model import (MultilayerPerceptrons, 
                            Module)

class DAR(Module):
    """
    Deep AutoRec
    """
    def __init__(self, io_dim: int,
                 hidden_dims:list = [512, 256, 128], 
                 e_activations:Union[str, list] = "relu", 
                 e_dropouts:Union[float, list] = 0,
                 d_activations:Union[str, list] = "relu", 
                 d_dropouts:Union[float, list] = 0,
                 latent_dropout: int = 0.5,
                 dense_refeeding = 1,
                 batch_norm:bool = True,
                 lr:float = 1e-3,
                 weight_decay:float = 1e-3,
                 criterion:F  = F.mse_loss):
        """
        Hyperparameters setting.

        :param io_dim (int): Input/Output dimension.

        :param hidden_dims (list): List of number of hidden nodes for encoder and decoder (in reverse). For example, hidden_dims [200, 100, 50] = encoder [200, 100, 50], decoder [50, 100, 200]. Default: [400, 400, 400]

        :param activations (str/list): List of activation functions. If type str, then the activation will be repeated len(hidden_dims) times in a list. Default: "relu"

        :param dropouts (float/list): List of dropouts. If type float, then the dropout will be repeated len(hidden_dims) times in a list. Default: 0.5

        :batch_norm (bool): If True, apply batch normalization on every hidden layer. Default: True

        :param lr (float): Learning rate. Default: 1e-3

        :param weight_decay (float): L2 regularization weight decap: Default: 1e-3

        :param criterion (F): Objective function. Default: F.mse_loss
        """
        super().__init__(lr, weight_decay, criterion)
        self.dense_refeeding = dense_refeeding

        # Encoder layer
        self.encoder = MultilayerPerceptrons(io_dim,
                                             hidden_dims, 
                                             e_activations, 
                                             e_dropouts,
                                             None,
                                             batch_norm)

        # Dropout layer in latent space
        self.dropout = torch.nn.Dropout(latent_dropout)

        # Decoder layer 
        self.decoder = MultilayerPerceptrons(hidden_dims[-1],
                                             [*hidden_dims[:-1][::-1], io_dim], 
                                             d_activations, 
                                             d_dropouts,
                                             None,
                                             batch_norm)
                                            
    def forward(self, x):
        for i in range(self.dense_refeeding):
            x = self.decoder(self.dropout(self.encoder(x)))

        return x