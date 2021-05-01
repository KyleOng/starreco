from typing import Union

import torch
import torch.nn.functional as F

from .module import BaseModule
from .layer import MultilayerPerceptrons

class SDAEmodel(torch.nn.Module):
    def __init__(self, 
                 input_output_dim:int,
                 feature_dim:int,
                 hidden_dims:list, 
                 e_activations:Union[str, list], 
                 d_activations:Union[str, list], 
                 e_dropouts:Union[int, float, list],
                 d_dropouts:Union[int, float, list],
                 latent_dropout:float,
                 batch_norm:bool,
                 noise_factor:Union[int, float],
                 mean:Union[int, float],
                 std:Union[int, float],
                 feature_all:bool,
                 noise_all:bool,
                 dense_refeeding:int):
        super().__init__()
        self.dense_refeeding = dense_refeeding
        self.noise_factor = noise_factor
        self.mean = mean
        self.std = std
        self.feature_all = feature_all
        self.noise_all = noise_all

        if self.feature_all:
            encoder_extra_input_dims = decoder_extra_input_dims = feature_dim
        else:
            encoder_extra_input_dims = [feature_dim] + [0] * len(hidden_dims)
            decoder_extra_input_dims = 0

        # Encoder layer
        self.encoder = MultilayerPerceptrons(input_dim = input_output_dim,
                                             hidden_dims = hidden_dims, 
                                             activations = e_activations, 
                                             dropouts = e_dropouts,
                                             batch_norm = batch_norm,
                                             remove_last_dropout = True,
                                             remove_last_batch_norm = False,
                                             output_layer = None,
                                             extra_input_dims = encoder_extra_input_dims,
                                             module_type = "modulelist")

        # Dropout layer in latent space
        if latent_dropout:
            self.dropout = torch.nn.Dropout(latent_dropout)

        # Decoder layer 
        self.decoder = MultilayerPerceptrons(input_dim = hidden_dims[-1],
                                             hidden_dims = [*hidden_dims[:-1][::-1], input_output_dim], 
                                             activations = d_activations, 
                                             dropouts = d_dropouts,
                                             batch_norm = batch_norm,
                                             remove_last_dropout = True,
                                             remove_last_batch_norm = True,
                                             output_layer = None,
                                             extra_input_dims = decoder_extra_input_dims,
                                             module_type = "modulelist")

    def add_noise(self, x):
        return x + self.noise_factor * torch.randn(x.size()).to(x.device) * self.std + self.mean

    def encode(self, x, feature = None):
        for i, module in enumerate(self.encoder.mlp):

            if self.training:
                if not(not self.noise_all and i):
                    x = self.add_noise(x)

            if type(module) == torch.nn.Linear and \
            feature is not None and \
            not(not self.feature_all and i):
                concat = torch.cat([x, feature], dim = 1)
                x = module(concat)
            else:
                x = module(x)

        return x

    def decode(self, x, feature = None):
        for i, module in enumerate(self.decoder.mlp):

            if self.training:
                if self.noise_all:
                    x = self.add_noise(x)

            if type(module) == torch.nn.Linear and \
            feature is not None and \
            self.feature_all:
                concat = torch.cat([x, feature], dim = 1)
                x = module(concat)
            else:
                x = module(x)

        return x

    def forward(self, x, feature = None):
        # Perform dense refeeding N times in training mode, else feed only once
        if self.training:
            dense_refeeding = self.dense_refeeding
        else:
            dense_refeeding = 1

        for i in range(dense_refeeding):
            x = self.encode(x, feature)
            x = self.dropout(x)
            x = self.decode(x, feature)
        return x


class SDAE(BaseModule):
    def __init__(self, 
                 input_output_dim:int,
                 feature_dim:int = 0,
                 hidden_dims:list = [512, 256, 128], 
                 e_activations:Union[str, list] = "relu", 
                 d_activations:Union[str, list] = "relu", 
                 e_dropouts:Union[int, float, list] = 0,
                 d_dropouts:Union[int, float, list] = 0,
                 latent_dropout:float = 0.5,
                 batch_norm:bool = True,
                 noise_factor:Union[int, float] = 0.3,
                 mean:Union[int, float] = 0,
                 std:Union[int, float] = 1,
                 dense_refeeding:int = 1,
                 feature_all:bool = True,
                 noise_all:bool = True,
                 lr:float = 1e-3,
                 weight_decay:float = 1e-3,
                 criterion:F = F.mse_loss):
        super().__init__(lr, weight_decay, criterion)
        self.model = SDAEmodel(input_output_dim, feature_dim, hidden_dims, e_activations, d_activations, e_dropouts, d_dropouts, latent_dropout, batch_norm, noise_factor, mean, std, feature_all, noise_all, dense_refeeding)
        self.save_hyperparameters()

    def forward(self, x, feature = None):
        return self.model.forward(x, feature)
        
        