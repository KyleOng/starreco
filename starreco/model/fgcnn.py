import math

import numpy as np
import torch
import torch.nn.functional as F

from .module import BaseModule
from .layers import MultilayerPerceptrons, ActivationFunction


class FGCNN(BaseModule):
    """
    Feature Generation Convolutional Neural Network.
    """

    def __init__(self,
                 user_features_split:list,
                 item_features_split:list,
                 field_dims:list,
                 embed_dim:int = 40,
                 c_filter_sizes:list = [6, 8, 10, 12],
                 kernel_size:int = 100,
                 pooling_size:int = 2,
                 r_filter_size:int = 3,
                 activation:str = "tanh",
                 lr:float = 1e-3,
                 weight_decay:float = 1e-6,
                 criterion = F.mse_loss
                 ):
        #(torch.tensor([0,1,0,1,0]) == 1.).nonzero().flatten().shape[0] == 2
        super().__init__(lr, weight_decay, criterion)