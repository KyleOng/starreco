from typing import Union

import torch
import torch.nn.functional as F

from .fm import AbstractFM
from .layers import MultilayerPerceptrons

# Done
class WDL(AbstractFM):
    """
    Wide and Deep Learning.

    - field_dims (list): List of features dimensions.
    - embed_dim (int): Embedding dimension. Default: 10.
    - hidden_dims (list): List of numbers of neurons across the hidden layers. Default: [400, 400, 400].
    - activations (str/list): List of activation functions. Default: relu.
    - dropouts (int/float/list): List of dropout values. Default: 0.5.
    - batch_norm (bool): If True, apply batch normalization in every layer. Batch normalization is applied between activation and dropout layer. Default: True.
    - lr (float): Learning rate. Default: 1e-3.
    - l2_lambda (float): L2 regularization rate. Default: 1e-3.
    - criterion (F): Criterion or objective or loss function. Default: F.mse_loss.
    """

    def __init__(self, field_dims:list, 
                 embed_dim:int = 10, 
                 hidden_dims:list = [400, 400, 400], 
                 activations:Union[str, list]  = "relu", 
                 dropouts:Union[int, float, list] = 0.5,
                 batch_norm:bool = True, 
                 lr:float = 1e-3,
                 weight_decay:float = 1e-3,
                 criterion:F = F.mse_loss):

        super().__init__(field_dims, embed_dim, lr, weight_decay, criterion)
        self.save_hyperparameters()

        # Multilayer Perceptrons
        self.net = MultilayerPerceptrons(input_dim = len(field_dims) * embed_dim,
                                         hidden_dims = hidden_dims, 
                                         activations = activations, 
                                         dropouts = dropouts,
                                         output_layer = "relu",
                                         batch_norm = batch_norm)

    def forward(self, x):
        # Generate embeddings
        embed_x = self.embedding(x.int())

        # Prediction
        linear = self.linear(x.int()) 
        net = self.net(torch.flatten(embed_x, start_dim = 1)) 
        y = linear + net 

        return y