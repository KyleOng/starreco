from typing import Union

import torch

from .module import BaseModule
from .layers import FeaturesEmbedding, StackedDenoisingAutoEncoder
from ..evaluation import masked_mse_loss

# Done
class CDAE(BaseModule):
    """
    Collaborative Denoising Autoencoder.

    - input_output_dim (int): Number of neurons in the input and output layer.
    - hidden_dim (int): Number of neurons in the hidden latent space. Default: 256.
    - activation (str): Activation function. Default: "relu".
    - noise_rate (int/float): Rate/Percentage of noises to be added to the input. Noise is not applied to extra input neurons. Noise is only applied during training only. Default: 1.
    - noise_factor (int/float): Noise factor. Default: 0.3
    - noise_all (bool): If True, noise are added to input and hidden layers, else only to input layer. Default: False.
    - mean (int/float): Gaussian noise mean. Default: 0.
    - std (int/float): Gaussian noise standard deviation: 1.
    - lr (float): Learning rate. Default: 1e-3.
    - l2_lambda (float): L2 regularization rate. Default: 1e-3.
    - criterion: Criterion or objective or loss function. Default: masked_mse_loss
    """

    def __init__(self, 
                 input_output_dim:int,
                 hidden_dim:int = 256, 
                 activation:str = "relu", 
                 dropout:float = 0.5,
                 batch_norm:bool = True,
                 noise_rate:Union[int, float] = 1,
                 noise_factor:Union[int, float] = 0.3,
                 noise_all:bool = True,
                 mean:Union[int, float] = 0,
                 std:Union[int, float] = 1,
                 lr:float = 1e-3,
                 weight_decay:float = 1e-3,
                 criterion = masked_mse_loss):

        super().__init__(lr, weight_decay, criterion)
        self.save_hyperparameters()

        self.features_embedding = FeaturesEmbedding([input_output_dim], hidden_dim)

        self.sdae = StackedDenoisingAutoEncoder(input_output_dim = input_output_dim, 
                                                hidden_dims = [hidden_dim], 
                                                e_activations = activation, 
                                                d_activations = "relu",
                                                e_dropouts = 0,
                                                d_dropouts = 0,
                                                dropout = dropout,
                                                batch_norm = batch_norm,
                                                extra_input_dims = 0,
                                                extra_output_dims = 0,
                                                noise_rate = noise_rate,
                                                noise_factor = noise_factor,
                                                noise_all = noise_all, 
                                                mean = mean,
                                                std = std)

    def forward(self, x, ids):
        # Add feature node embeddings to latent repsentation
        embed_ids = self.features_embedding(ids.int())
        embed_ids = torch.flatten(embed_ids, start_dim = 1)
        x = self.sdae.decode(self.sdae.encode(x) + embed_ids)
        return x  

