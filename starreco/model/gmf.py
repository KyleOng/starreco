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
                 lr:float = 1e-2,
                 weight_decay:float = 1e-6,
                 criterion:F = F.mse_loss,
                 save_hyperparameters:bool = True):
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

        # Singlelayer perceptrons
        self.nn = MultilayerPerceptrons(input_dim = embed_dim, 
                                        output_layer = "relu")

        # Save hyperparameters to checkpoint
        if save_hyperparameters:
            self.save_hyperparameters()
    
    def forward(self, x):
        """
        Perform operations.

        :x (torch.tensor): Input tensors of shape (batch_size, 2) user and item.

        :return (torch.tensor): Output prediction tensors of shape (batch_size, 1)
        """
        # Generate embeddings
        x = self.embedding(x)
        # Seperate user (1st column) and item (2nd column) embeddings from generated embeddings
        user_embedding = x[:, 0]
        item_embedding = x[:, 1]

        # Element wise product between user and item embeddings
        product = user_embedding * item_embedding

        # Feed element wise product to generalized non-linear layer
        y = self.nn(product)

        return y
