import torch.nn.functional as F

from .hdar import HDAR

class HAR(HDAR):
    """
    Hybrid AutoRec
    """
    def __init__(self, io_dim:int, feature_dim:int,
                 latent_dim: int = 20, 
                 e_activations:str = "relu", 
                 d_activations:str = "relu", 
                 dropout: int = 0.5,
                 batch_norm:bool = True,
                 dense_refeeding = 1,
                 lr:float = 1e-3,
                 weight_decay:float = 1e-3,
                 criterion:F = F.mse_loss):
        """
        Hyperparameters setting.

        :param io_dim (int): Input/Output dimension.

        :param feature_dim (int): Feature dimension.

        :param latent_dim (list): Latent space dimension.

        :param e_activations (str/list): Activation functions for encoder layer. Default: "relu"

        :param d_activations (str/list): Activation functions for decoder layer. Default: "relu"

        :param dropout (float): Dropout within latent space. Default: 0.5

        :param dense_refeeding (int): Number of dense refeeding: Default: 1

        :param batch_norm (bool): If True, apply batch normalization on every hidden layer. Default: True

        :param lr (float): Learning rate. Default: 1e-3

        :param weight_decay (float): L2 regularization weight decap: Default: 1e-3

        :param criterion (F): Objective function. Default: F.mse_loss
        """

        super().__init__(io_dim, feature_dim, [latent_dim], e_activations, d_activations, dropout, dense_refeeding, batch_norm, lr, weight_decay, criterion) 
