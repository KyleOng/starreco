from typing import Union

import torch
import torch.nn.functional as F

from .module import BaseModule
from .layers import FeaturesEmbedding, FeaturesLinear, PairwiseInteraction, MultilayerPerceptrons

# Done
class NFM(BaseModule):
    """
    Neural Factorization Machine.

    - field_dims (list): List of features dimensions.
    - embed_dim (int): Embedding dimension. Default: 64.
    - bi_dropout (int/float): Dropout value for Bi-interaction layer. Default: 0.2.
    - hidden_dims (list): List of numbers of neurons across the hidden layers. Default: [1024, 512, 256].
    - activations (str/list): List of activation functions. Default: "relu".
    - dropouts (int/float/list): List of dropout values. Default: 0.5.
    - batch_norm (bool): If True, apply batch normalization in every layer. Batch normalization is applied between activation and dropout layer. Default: True.
    - lr (float): Learning rate. Default: 1e-3.
    - weight_decay (float): L2 regularization rate. Default: 1e-3.
    - criterion: Criterion or objective or loss function. Default: F.mse_loss.
    """
    def __init__(self, field_dims:list, 
                 embed_dim:int = 64, 
                 bi_dropout:Union[int, float] = 0.2,
                 hidden_dims:list = [1024, 512, 256], 
                 activations:Union[str, list]  = "relu", 
                 dropouts:Union[int, float, list] = 0.5,
                 batch_norm:bool = True, 
                 lr:float = 1e-3,
                 weight_decay:float = 1e-3,
                 criterion = F.mse_loss):
        super().__init__(lr, weight_decay, criterion)
        self.save_hyperparameters()

        # Embedding layer
        self.features_embedding = FeaturesEmbedding(field_dims, embed_dim)

        # Linear layer
        self.features_linear = FeaturesLinear(field_dims)

        # Bi-interaction layer
        bi_interaction_blocks = [PairwiseInteraction(reduce_sum = False), torch.nn.BatchNorm1d(embed_dim)]
        if bi_dropout: 
            bi_interaction_blocks.append(torch.nn.Dropout(bi_dropout))
        self.bi_interaction = torch.nn.Sequential(*bi_interaction_blocks)

         # Multilayer Perceptrons
        self.net = MultilayerPerceptrons(input_dim = embed_dim,
                                         hidden_dims = hidden_dims, 
                                         activations = activations, 
                                         dropouts = dropouts,
                                         output_layer = "relu",
                                         batch_norm = batch_norm)

    def forward(self, x):
        # Linear regression 
        linear = self.features_linear(x.int()) 

        # Bi-interaction between embedding
        x_embed = self.features_embedding(x.int())
        cross_term = self.bi_interaction(x_embed)

        # Non linear on bi-interaction
        cross_net = self.net(cross_term)

        # Sum
        y =  linear + cross_net

        return y