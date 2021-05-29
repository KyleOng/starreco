import torch.nn.functional as F

from .mf import MF
from .layers import MultilayerPerceptrons

# Done
class GMF(MF):
    """
    Generalized Matrix Factorization.

    - field_dims (list): List of features dimensions.
    - embed_dim (int): Embedding dimension. Default: 8.
    - lr (float): Learning rate. Default: 1e-3.
    - l2_lambda (float): L2 regularization rate. Default: 1e-3.
    - criterion: Criterion or objective or loss function. Default: F.mse_loss.
    """

    def __init__(self, 
                 field_dims:list, 
                 embed_dim:int = 8, 
                 lr:float = 1e-3,
                 l2_lambda:float = 1e-6,
                 criterion = F.mse_loss):
        super().__init__(field_dims, embed_dim, lr, l2_lambda, criterion)
        self.save_hyperparameters()

        # Multilayer Perceptrons layer
        self.net = MultilayerPerceptrons(input_dim = embed_dim, 
                                         output_layer = "relu")

    def forward(self, x):
        # Element wise product between user and item embeddings
        x_embed = self.features_embedding(x.int())
        user_embed, item_embed = x_embed[:, 0], x_embed[:, 1]
        product = user_embed * item_embed
        
        # Non linear on element wise product
        y = self.net(product)

        return y        