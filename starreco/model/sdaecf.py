from typing import Union

from .module import BaseModule
from .layers import StackedDenoisingAutoEncoder
from ..evaluation import masked_mse_loss

# Done
class SDAECF(BaseModule):
    """
    Stacked Denoising AutoEncoder for Collaborative Filtering.

    - input_output_dim (int): Number of neurons in the input and output layer.
    - hidden_dims (list): List of number of neurons throughout the encoder and decoder (reverse list) hidden layers. Default: [512, 256, 128].
    - e_activations (str/list): List of activation functions in the encoder layers. Default: "relu".
    - d_activations (str/list): List of activation functions in the decoder layers. Default: "relu".
    - e_dropouts (int/float/list): List of dropout values in the encoder layers. Default: 0.
    - d_dropouts (int/float/list): List of dropout values in the decoder layers. Default: 0.
    - dropout (float): Dropout value in the latent space. Default: 0.5.
    - batch_norm (bool): If True, apply batch normalization in all hidden layers. Default: True.
    - noise_rate (int/float): Rate/Percentage of noises to be added to the input. Noise is not applied to extra input neurons. Noise is only applied during training only. Default: 1.
    - noise_factor (int/float): Noise factor. Default: 0.3.
    - noise_all (bool): If True, noise are added to inputs in all input and hidden layers, else only to input layer. Default: False.
    - mean (int/float): Gaussian noise mean. Default: 0.
    - std (int/float): Gaussian noise standard deviation: 0.5.
    - alpha (int/float): Trade off parameter for denoising loss. Default 1.2.
    - beta (int/float): Trade off parameter for reconstruction loss. Default 0.8.
    - lr (float): Learning rate. Default: 1e-3.
    - l2_lambda (float): L2 regularization rate. Default: 1e-3.
    - criterion: Criterion or objective or loss function. Default: masked_mse_loss
    """

    def __init__(self, 
                 input_output_dim:int,
                 hidden_dims:list = [512, 256, 128], 
                 e_activations:Union[str, list] = "relu", 
                 d_activations:Union[str, list] = "relu", 
                 e_dropouts:Union[int, float, list] = 0,
                 d_dropouts:Union[int, float, list] = 0,
                 dropout:float = 0.5,
                 batch_norm:bool = True,
                 noise_rate:Union[int, float] = 0.5,
                 noise_factor:Union[int, float] = 0.3,
                 noise_all:bool = True,
                 mean:Union[int, float] = 0,
                 std:Union[int, float] = 1,
                 alpha:Union[int, float] = 1.2,
                 beta:Union[int, float] = 0.8,
                 lr:float = 1e-3,
                 weight_decay:float = 1e-3,
                 criterion = masked_mse_loss):

        super().__init__(lr, weight_decay, criterion)
        self.save_hyperparameters()
        self.alpha = alpha
        self.beta = beta
        
        self.sdae = StackedDenoisingAutoEncoder(input_output_dim = input_output_dim, 
                                                hidden_dims = hidden_dims, 
                                                e_activations = e_activations, 
                                                d_activations = d_activations,
                                                e_dropouts = e_dropouts,
                                                d_dropouts = d_dropouts,
                                                dropout = dropout,
                                                batch_norm = batch_norm,
                                                extra_input_dims = 0,
                                                extra_output_dims = 0,
                                                noise_rate = noise_rate,
                                                noise_factor = noise_factor,
                                                noise_all = noise_all, 
                                                mean = mean,
                                                std = std)

    def forward(self, x):
        # Add feature node embeddings to latent repsentation
        return self.sdae(x)

    def backward_loss(self, *batch):
        """
        Custom backward loss.
        """
        if self.training:
            # Balance whether the network would focus on denoising the input (α) or reconstructing theinput
            xs = batch[:-1]
            y = batch[-1]
            y_hat = self.forward(*xs)

            # Denoising loss
            y_denoised = y * self.sdae.noise_masks[0]
            denoising_loss = self.alpha * self.criterion(y_hat, y_denoised)

            # Reconstruction loss
            y_reconstructed = y * ~self.sdae.noise_masks[0]
            reconstruction_loss = self.beta * self.criterion(y_hat, y_reconstructed)

            loss = denoising_loss + reconstruction_loss

            return loss
        else:
            return super().backward_loss(*batch)

