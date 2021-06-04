from typing import Union

from .module import BaseModule
from .layers import StackedDenoisingAutoEncoder
from ..evaluation import masked_mse_loss

# Done
class CFN(BaseModule):
    """
    Collaborative Filtering Network.

    - input_output_dim (int): Number of neurons in the input and output layer.
    - hidden_dims (list): List of number of neurons throughout the encoder and decoder (reverse list) hidden layers. Default: [512, 256, 128].
    - e_activations (str/list): List of activation functions in the encoder layers. Default: "relu".
    - d_activations (str/list): List of activation functions in the decoder layers. Default: "relu".
    - e_dropouts (int/float/list): List of dropout values in the encoder layers. Default: 0.
    - d_dropouts (int/float/list): List of dropout values in the decoder layers. Default: 0.
    - dropout (float): Dropout value in the latent space. Default: 0.5.
    - batch_norm (bool): If True, apply batch normalization in all hidden layers. Default: True.
    - extra_input_dims (int): Extra input neuron. Default: 0.
    - extra_input_all (bool): If True, extra input neurons are added to all input and hidden layers, else only to input layer. Default: False.
    """

    def __init__(self, 
                 input_output_dim:int,
                 feature_dim:int,
                 hidden_dims:list = [512, 256, 128], 
                 e_activations:Union[str, list] = "relu", 
                 d_activations:Union[str, list] = "relu", 
                 e_dropouts:Union[int, float, list] = 0,
                 d_dropouts:Union[int, float, list] = 0,
                 dropout:float = 0.5,
                 batch_norm:bool = True,
                 feature_input_all:bool = False,
                 lr:float = 1e-3,
                 weight_decay:float = 1e-3,
                 criterion = masked_mse_loss):
        super().__init__(lr, weight_decay, criterion)
        self.save_hyperparameters()

        if feature_input_all:
            extra_input_dims = feature_dim
        else:
            extra_input_dims = [feature_dim] + [0] * (len(hidden_dims) * 2 - 1)

        self.sdae = StackedDenoisingAutoEncoder(input_output_dim = input_output_dim, 
                                                hidden_dims = hidden_dims, 
                                                e_activations = e_activations, 
                                                d_activations = d_activations,
                                                e_dropouts = e_dropouts,
                                                d_dropouts = d_dropouts,
                                                dropout = dropout,
                                                batch_norm = batch_norm,
                                                extra_input_dims = extra_input_dims,
                                                extra_output_dims = 0,
                                                noise_factor = 0,
                                                noise_all = False, 
                                                mean = 0,
                                                std = 0)

    def forward(self, x, feature):
        return self.sdae(x, feature)

