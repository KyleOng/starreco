from typing import Union

import torch
import torch.nn.functional as F

from .wdl import WDL
from .layers import CompressedInteraction

# Done
class XDFM(WDL):
    """
    Neural Factorization Machine.

    - field_dims (list): List of features dimensions.
    - embed_dim (int): Embedding dimension. Default: 10.
    - hidden_dims (list): List of numbers of neurons across the hidden layers. Default: [400, 400, 400].
    - activations (str/list): List of activation functions. Default: "relu".
    - dropouts (int/float/list): List of dropout values. Default: 0.5.
    - cross_dims: List of numbers of neurons across cross interaction layers. Default: [200, 200]
    - cross_split_half: If True, convolution output across the cross interaction layers is splitted into half of the 1st dimension. Default: True.
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
                 cross_dims:list = [200, 200], 
                 cross_split_half:bool = True,
                 lr:float = 1e-3,
                 weight_decay:float = 1e-3,
                 criterion:F = F.mse_loss):
        super().__init__(field_dims, embed_dim, hidden_dims, activations, dropouts, batch_norm, lr, weight_decay, criterion)
        self.save_hyperparameters()

        # Compressed Interaction
        self.compressed_interaction = CompressedInteraction(len(field_dims), cross_dims, split_half = cross_split_half)

    def forward(self, x):
        # Linear regression
        linear = self.features_linear(x.int()) 

        # Compress interaction between embeddings
        x_embed = self.features_embedding(x.int())
        compress_interaction = self.compressed_interaction(x_embed)

        # Non linear on concatenated embeddings
        embed_concat = torch.flatten(x_embed, start_dim = 1)
        net = self.net(embed_concat) 

        # Sum
        y = linear + compress_interaction + net 

        return y