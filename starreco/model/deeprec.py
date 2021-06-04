from typing import Union

from .module import BaseModule
from .layers import StackedDenoisingAutoEncoder
from ..evaluation import masked_mse_loss

# Done
class DeepRec(BaseModule):
    """
    DeepRec.

    - input_output_dim (int): Number of neurons in the input and output layer.
    - hidden_dims (list): List of number of neurons throughout the encoder and decoder (reverse list) hidden layers. Default: [512, 256, 128].
    - e_activations (str/list): List of activation functions in the encoder layers. Default: "relu".
    - d_activations (str/list): List of activation functions in the decoder layers. Default: "relu".
    - lr (float): Learning rate. Default: 1e-3.
    - l2_lambda (float): L2 regularization rate. Default: 1e-3.
    - criterion : Criterion or objective or loss function. Default: masked_mse_loss
    """

    def __init__(self, 
                 input_output_dim:int,
                 hidden_dims:list = [512, 256, 128], 
                 e_activations:Union[str, list] = "relu", 
                 d_activations:Union[str, list] = "relu", 
                 dropout:float = 0.5,
                 batch_norm:bool = True,
                 lr:float = 1e-3,
                 weight_decay:float = 1e-3,
                 criterion = masked_mse_loss):
        super().__init__(lr, weight_decay, criterion)
        self.save_hyperparameters()

        self.sdae = StackedDenoisingAutoEncoder(input_output_dim = input_output_dim, 
                                                hidden_dims = hidden_dims, 
                                                e_activations = e_activations, 
                                                d_activations = d_activations,
                                                e_dropouts = 0,
                                                d_dropouts = 0,
                                                dropout = dropout,
                                                batch_norm = batch_norm,
                                                extra_input_dims = 0,
                                                extra_output_dims = 0,
                                                noise_factor = 0,
                                                noise_all = False, 
                                                mean = 0,
                                                std = 0)

    def forward(self, x):
        return self.sdae(x)

