from typing import Union

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .module import BaseModule
from .layers import DocumentContextAnalysis

# Done
class CMF(BaseModule):
    """
    Convolutional Matrix Factorization.

    - user_dim (int): User dimension.
    - word_dim (int): Word/vocabulary dimension.
    - max_len (int): Max sentence length.
    - user_embed_dim (int): User embedding dimension.
    - word_embed_dim (int): Word/vocabulary embedding dimension.
    - filter_size (int): Convolution filter/depth/channel size. Default: 100.
    - kernel_size (int): Convolution square-window size or convolving square-kernel size. Default: 3
    - c_activation (str): Activation function applied across the convolution layers. Default: "relu".
    - c_dropout (int/float): Convolution dropout rate. Default: 0.
    - fc_dim (int): Fully connected layer dimension. Default: 200.
    - fc_activation (str): Fully connected layer activation function. Default: "tanh".
    - fc_dropout (int/float): Fully connected layer dropout rate. Default: 0.
    - batch_norm (bool): If True, apply batch normalization during convolutions and fully connected layer. Batch normalization is applied between activation and dropout layer across the convolution layers. Default: True.
    - lr (float): Learning rate. Default: 1e-3.
    - weight_decay (float): L2 regularization rate. Default: 1e-3.
    - criterion: Criterion or objective or loss function. Default: F.mse_loss.

    Note: CNN is used to address the problem of varying sentences length by taking the maximum (pooling) of the convoluted sentence embeddings.
    """

    def __init__(self, 
                 user_dim:int,
                 word_dim:int,
                 max_len:int, 
                 user_embed_dim:int = 50,
                 word_embed_dim:int = 200,
                 filter_size:list = 100,
                 kernel_size:int = 3,
                 c_activation:str = "relu",
                 c_dropout:Union[int, float] = 0,
                 fc_dim:int = 200,
                 fc_activation:str = "tanh", 
                 fc_dropout:Union[int, float] = 0.2,
                 batch_norm:bool = True,
                 lr:float = 1e-3,
                 weight_decay:float = 1e-6,
                 criterion = F.mse_loss):
        super().__init__(lr, weight_decay, criterion)
        self.save_hyperparameters()

        # User embedding layer
        self.user_embedding = torch.nn.Embedding(user_dim, user_embed_dim)
        
        # Document context analysis
        self.cnn = DocumentContextAnalysis(word_dim, max_len, user_embed_dim, word_embed_dim, filter_size, kernel_size, c_activation, c_dropout, fc_dim, fc_activation, fc_dropout, batch_norm)

    def forward(self, x, _, document_x):
        # Get user embeddings
        user_x = x[:,0].int()
        user_embed = self.user_embedding(user_x)

        # Get context aware document latent vector
        item_embed = self.cnn(document_x)

        # Dot product between user and item embeddings
        dot = torch.sum(user_embed * item_embed, dim = 1)
        
        # Reshape to match target shape
        y = dot.view(dot.shape[0], -1)

        return y
