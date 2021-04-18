from typing import Union

import torch
import torch.nn.functional as F

from .module import BaseModule
from .layer import MultilayerPerceptrons

class SDAE(torch.nn.Module):
    def __init__(self, io_dim:int,
                 hidden_dims:list = [512, 256, 128], 
                 e_activations:Union[str, list] = "relu", 
                 d_activations:Union[str, list] = "relu", 
                 dropout:float = 0.5,
                 dense_refeeding:int = 1,
                 batch_norm:bool = True):
        super().__init__()
        self.dense_refeeding = dense_refeeding

        # Encoder layer
        self.encoder = MultilayerPerceptrons(input_dim = io_dim,
                                             hidden_dims = hidden_dims, 
                                             activations = e_activations, 
                                             dropouts = 0,
                                             output_layer = None,
                                             batch_norm = batch_norm)

        # Dropout layer in latent space
        self.dropout = torch.nn.Dropout(dropout)

        # Decoder layer 
        self.decoder = MultilayerPerceptrons(input_dim = hidden_dims[-1],
                                             hidden_dims = [*hidden_dims[:-1][::-1], io_dim], 
                                             activations = d_activations, 
                                             dropouts = 0,
                                             apply_last_bndp = False,
                                             output_layer = None,
                                             batch_norm = batch_norm)

    def encode(self, x):
        return self.encoder(x)

    def forward(self, x):
        # Perform dense refeeding N times in training mode, else feed only once
        if self.training:
            dense_refeeding = self.dense_refeeding
        else:
            dense_refeeding = 1

        for i in range(dense_refeeding):
            x = self.decoder(self.dropout(self.encode(x)))

        return x


class SDAEModule(BaseModule):
    def __init__(self, io_dim:int,
                 hidden_dims:list = [512, 256, 128], 
                 e_activations:Union[str, list] = "relu", 
                 d_activations:Union[str, list] = "relu", 
                 dropout:float = 0.5,
                 dense_refeeding:int = 1,
                 batch_norm:bool = True,
                 lr:float = 1e-3,
                 weight_decay:float = 1e-3,
                 criterion:F = F.mse_loss):
        super().__init__(lr, weight_decay, criterion)
        self.model = SDAE(io_dim, hidden_dims, e_activations, d_activations, dropout, dense_refeeding, batch_norm)
        self.save_hyperparameters()
        