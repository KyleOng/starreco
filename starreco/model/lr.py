import torch.nn.functional as F

from .module import BaseModule
from .layers import FeaturesLinear

# Done
class LR(BaseModule):
    """
    Linear Regression.

    - field_dims (list): List of features dimensions. 
    - embed_dim (int): Embedding dimension. Default: 8.
    - lr (float): Learning rate. Default: 1e-3.
    - l2_lambda (float): L2 regularization rate. Default: 1e-3.
    - criterion (F): Criterion or objective or loss function. Default: F.mse_loss.
    """

    def __init__(self, 
                 field_dims:list, 
                 lr:float = 1e-3,
                 weight_decay:float = 1e-3,
                 criterion:F = F.mse_loss):
        super().__init__(lr, weight_decay, criterion)
        self.save_hyperparameters()

        # Linear layer
        self.linear = FeaturesLinear(field_dims)

    def forward(self, x):
        
        # Prediction
        y = self.linear(x.int())

        return y