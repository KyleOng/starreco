import torch
import torch.nn.functional as F

from .oncf import ONCF
from .layers import MultilayerPerceptrons

# Done
class CNNDCF(ONCF):
    """
    Outer Product-based Neural Collaborative Filtering.

    - field_dims (list): List of features dimensions.
    - embed_dim (int): Embedding dimension.
    - filter_size (int): Convolution filter/depth/channel size. Default: 32.
    - kernel_size (int): Convolution square-window size or convolving square-kernel size. Default: 2
    - stride (int): Convolution stride. Default: 2.
    - activation (str): Activation function applied across the convolution layers. Default: relu.
    - batch_norm (bool): If True, apply batch normalization after c. Batch normalization is applied between activation and dropout layer across the convolution layers. Default: True.
    - lr (float): Learning rate. Default: 1e-3.
    - l2_lambda (float): L2 regularization rate. Default: 1e-3.
    - criterion (F): Criterion or objective or loss function. Default: F.mse_loss.  
    """

    def __init__(self, field_dims:list, 
                 embed_dim:int = 32, #or 64
                 filter_size:int = 32, #or 64
                 kernel_size:int = 2, 
                 stride:int = 2,
                 activation:str = "relu", 
                 dropout:bool = 0.5,
                 batch_norm:bool = True, 
                 lr:float = 1e-3,
                 weight_decay:float = 1e-3,
                 criterion:F = F.mse_loss):
        super().__init__(field_dims, embed_dim, filter_size, kernel_size, stride, activation, dropout, batch_norm, lr, weight_decay, criterion)
        self.save_hyperparameters()

        # Residual connection layer
        self.residual = MultilayerPerceptrons(input_dim = embed_dim ** 2,
                                              hidden_dims = [embed_dim ** 2],
                                              remove_last_dropout = True,
                                              remove_last_batch_norm = True,
                                              output_layer = None)

    def forward(self, x):
        # Generate embeddings
        user_embed, item_embed = self.user_item_embeddings(x)

        # Outer product between user and items embeddings
        outer = torch.bmm(user_embed.unsqueeze(2), item_embed.unsqueeze(1))

        # Residual connection
        outer_flatten = torch.flatten(outer, start_dim = 1)
        residual = self.residual(outer_flatten)
        residual = residual.view(outer.shape)
        outer_residual = outer + residual

        # Unsqueeze outer product so that each matrix contain single depth for convolution
        outer_residual = torch.unsqueeze(outer_residual, 1)
        
        # Prediction
        y = self.cnn(outer_residual)

        return y