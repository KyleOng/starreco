from typing import Union

import torch.nn.functional as F

from .dar import DAR

class AR(DAR):
    """
    AutoRec
    """
    def __init__(self, io_dim: int,
                 hidden_dim: int = 20, 
                 e_activations:Union[str, list] = "relu", 
                 e_dropouts:Union[float, list] = 0,
                 d_activations:Union[str, list] = "relu", 
                 d_dropouts:Union[float, list] = 0,
                 latent_dropout: int = 0.5,
                 batch_norm:bool = True,
                 dense_refeeding = 1,
                 lr:float = 1e-3,
                 weight_decay:float = 1e-3,
                 criterion:F  = F.mse_loss):

        super().__init__(io_dim, [hidden_dim], e_activations, e_dropouts, d_activations, 
        d_dropouts, latent_dropout, dense_refeeding, batch_norm, lr, weight_decay, criterion) 