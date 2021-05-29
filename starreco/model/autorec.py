from .deeprec import DeepRec
from ..evaluation import masked_mse_loss

# Done
class AutoRec(DeepRec):
    """
    AutoRec.

    - input_output_dim (int): Number of neurons in the input and output layer.
    - hidden_dims (int): List of number of neurons throughout the encoder and decoder (reverse list) hidden layers. Default: 256.
    - e_activations (str): List of activation functions in the encoder layers. Default: "relu".
    - d_activations (str): List of activation functions in the decoder layers. Default: "relu".
    - lr (float): Learning rate. Default: 1e-3.
    - l2_lambda (float): L2 regularization rate. Default: 1e-3.
    - criterion: Criterion or objective or loss function. Default: masked_mse_loss
    """

    def __init__(self, 
                 input_output_dim:int,
                 hidden_dim:int = 256, 
                 e_activation:str = "relu", 
                 d_activation:str = "relu", 
                 dropout:float = 0.5,
                 batch_norm:bool = True,
                 lr:float = 1e-3,
                 weight_decay:float = 1e-3,
                 criterion = masked_mse_loss):
        super().__init__(input_output_dim, [hidden_dim], e_activation, d_activation, dropout, batch_norm, lr, weight_decay, criterion)
        self.save_hyperparameters()