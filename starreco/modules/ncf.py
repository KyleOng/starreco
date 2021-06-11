from typing import Union

import torch
import torch.nn.functional as F

from .mf import MF
from .layers import MultilayerPerceptrons

# Done
class NCF(MF):
    """
    Neural Collaborative Filtering.

    - field_dims (list): List of features dimensions.
    - embed_dim (int): Embedding dimension.
    - hidden_dims (list): List of numbers of neurons throughout the hidden layers. Default: [32,16,8].
    - activations (str/list): List of activation functions. Default: "relu".
    - dropouts (int/float/list): List of dropout values. Default: 0.5.
    - batch_norm (bool): If True, apply batch normalization in every layer. Batch normalization is applied between activation and dropout layer. Default: True.
    - lr (float): Learning rate. Default: 1e-3.
    - weight_decay (float): L2 regularization rate. Default: 1e-3.
    - criterion: Criterion or objective or loss function. Default: F.mse_loss.  
    """

    def __init__(self, 
                 field_dims:list, 
                 embed_dim:int = 8,
                 hidden_dims:list = [32, 16, 8], 
                 activations:Union[str, list] = "relu", 
                 dropouts:Union[int, float, list] = 0.5, 
                 batch_norm:bool = True,
                 lr:float = 1e-3,
                 weight_decay:float = 1e-3,
                 criterion = F.mse_loss):
        super().__init__(field_dims, embed_dim, lr, weight_decay, criterion)
        self.save_hyperparameters()

        # Multilayer Perceptrons layer
        self.net = MultilayerPerceptrons(input_dim = 2 * embed_dim, # strictly only 2
                                         hidden_dims = hidden_dims, 
                                         activations = activations, 
                                         dropouts = dropouts,
                                         output_layer = "relu",
                                         batch_norm = batch_norm)

    def forward(self, x):
        # Concatenate user and item embeddings
        x_embed = self.features_embedding(x.int())
        embed_concat = torch.flatten(x_embed, start_dim = 1)

        # Non linear on concatenated user and item embeddings
        y = self.net(embed_concat)

        return y