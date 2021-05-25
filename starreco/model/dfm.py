from starreco.model.module import BaseModule
from typing import Union

import torch
import torch.nn.functional as F

from .module import BaseModule
from .layer import FeaturesEmbedding, FeaturesLinear, PairwiseInteraction, MultilayerPerceptrons

# Done
class DFM(BaseModule):
    """
    Neural Factorization Machine.

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

        super().__init__(lr, weight_decay, criterion)
        self.save_hyperparameters()

        if type(dropouts) == float or type(dropouts) == int:
            dropouts = [dropouts] * len(hidden_dims)

        # Embedding layer
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)

        # Linear layer
        self.linear = FeaturesLinear(field_dims)

        # Pairwise interaction
        self.pairwise_interaction = PairwiseInteraction()

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
        y = self.linear(x.int()) + self.pairwise_interaction(embed_x) + self.net(torch.flatten(embed_x, start_dim = 1)) 

        return y