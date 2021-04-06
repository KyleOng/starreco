from typing import Union

import torch
import torch.nn.functional as F

from .hdae import HDAE
from starreco.model import MultilayerPerceptrons

class ASAE(HDAE):
    """
    Additional Stacked/Deep Autoencoder
    """
    def __init__(self, io_dim:int, feature_dim:int,
                 feature_concat_all:bool = True,
                 hidden_dims:list = [512, 256, 128], 
                 e_activations:Union[str, list] = "relu", 
                 d_activations:Union[str, list] = "relu", 
                 dropout:float = 0.5,
                 dense_refeeding:int = 1,
                 batch_norm:bool = True,
                 lr:float = 1e-3,
                 weight_decay:float = 1e-3,
                 criterion:F = F.mse_loss,
                 save_hyperparameters:bool = True):
        """
        Hyperparameters setting.

        :param io_dim (int): Input/Output dimension.

        :param feature_dim (int): Feature (side information) dimension.

        :param feature_concat_all (bool): If True concat feature on input layer and all hidden layers, else concat feature on input layer only. Default: True

        :param hidden_dims (list): List of number of hidden nodes for encoder and decoder (in reverse). For example, hidden_dims [200, 100, 50] = encoder [io_dim, 200, 100, 50], decoder [50, 100, 200, io_dim]. Default: [512, 256, 128]

        :param e_activations (str/list): List of activation functions for encoder layer. If type str, then the activation will be repeated len(hidden_dims) times in a list. Default: "relu"

        :param d_activations (str/list): List of activation functions for decoder layer. If type str, then the activation will be repeated len(hidden_dims) times in a list. Default: "relu"

        :param dropout (float): Dropout within latent space. Default: 0.5

        :param dense_refeeding (int): Number of dense refeeding: Default: 1

        :param batch_norm (bool): If True, apply batch normalization on every hidden layer. Default: True

        :param lr (float): Learning rate. Default: 1e-3

        :param weight_decay (float): L2 regularization weight decay: Default: 1e-3

        :param criterion (F): Objective function. Default: F.mse_loss
        """

        super().__init__(io_dim, feature_dim, feature_concat_all, hidden_dims, e_activations, d_activations, dropout, dense_refeeding, batch_norm, lr, weight_decay, criterion) 

        if not feature_concat_all:
            extra_nodes_in = 0
        else:
            extra_nodes_in = feature_dim

        # Decoder layer 
        self.decoder = MultilayerPerceptrons(input_dim = hidden_dims[-1],
                                             hidden_dims = [*hidden_dims[:-1][::-1]], 
                                             activations = d_activations, 
                                             dropouts = 0,
                                             apply_last_bndp = True,
                                             output_layer = None,
                                             batch_norm = batch_norm,
                                             extra_nodes_in = extra_nodes_in,
                                             module_type = "modulelist")  

        self.decoder_x1 = MultilayerPerceptrons(input_dim = hidden_dims[0],
                                                hidden_dims = [io_dim],
                                                activations = d_activations,
                                                dropouts = 0,
                                                apply_last_bndp = False,
                                                output_layer = None,
                                                batch_norm = batch_norm,
                                                extra_nodes_in = extra_nodes_in)

        self.decoder_x2 = MultilayerPerceptrons(input_dim = hidden_dims[0],
                                                hidden_dims = [feature_dim],
                                                activations = d_activations,
                                                dropouts = 0,
                                                apply_last_bndp = False,
                                                output_layer = None,
                                                batch_norm = batch_norm,
                                                extra_nodes_in = extra_nodes_in)        

    def forward(self, x):
        # Perform dense refeeding N times in training mode, else feed only once
        if self.training:
            dense_refeeding = self.dense_refeeding
        else:
            dense_refeeding = 1

        for i in range(dense_refeeding):
            feature = x[:, :self.feature_dim]
            x = x[:, self.feature_dim:]
            x = self.encode(x, feature)    
            x = self.dropout(x)
            for module in self.decoder.mlp:
                if type(module) == torch.nn.Linear:
                    x = module(torch.cat([x, feature], dim = 1))
                else:
                    x = module(x)
            x1 = self.decoder_x1(torch.cat([x, feature], dim = 1))
            x2 = self.decoder_x2(torch.cat([x, feature], dim = 1))
            x = torch.cat([x1, x2], dim = 1)

        return x





        
