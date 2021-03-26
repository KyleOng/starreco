import torch
import torch.nn.functional as F

from starreco.model import (FeaturesEmbedding, 
                            MultilayerPerceptrons,
                            Module)

class GMF(Module):
    """
    Generalized Matrix Factorization
    """
    def __init__(self, feature_dims:list, 
                 embed_dim:int = 8, 
                 activation:str = "relu",
                 lr:float = 1e-2,
                 weight_decay:float = 1e-6,
                 criterion:F = F.mse_loss):
        """
        Hyperparameters setting.

        :param feature_dims (list): List of feature dimensions. 

        :param embed_dim (int): Embeddings dimensions. Default: 8

        :param activation (str): Activation Function. Default: "relu".

        :param lr (float): Learning rate. Default: 1e-3

        :param weight_decay (float): L2 regularization weight decay: Default: 1e-3

        :param criterion (F): Objective function. Default: F.mse_loss
        """
        super().__init__(lr, weight_decay, criterion)

        # Embedding layer
        self.embedding = FeaturesEmbedding(feature_dims, embed_dim)

        # Multilayer perceptrons
        self.mlp = MultilayerPerceptrons(input_dim = embed_dim, 
                                         activations = activation, 
                                         output_layer = "relu")
    

    def forward(self, x):
        """
        Perform operations.

        :x (torch.tensor): Input tensors of shape (batch_size, 2) user and item.

        :return (torch.tensor): Output prediction tensors of shape (batch_size, 1)
        """
        # Generate embeddings
        x = self.embedding(x)
        user_embedding = x[:, 0]
        item_embedding = x[:, 1]

        # Element wise product between embeddings
        product = user_embedding * item_embedding

        # Prediction
        return self.mlp(product)
