import numpy as np
import torch

class FeaturesEmbedding(torch.nn.Module):
    def __init__(self, feature_dims: list, embed_dim: int):
        """
        Embedd Features.

        :param feature_dims: list. List of feature dimension. Each feature contains
        a total number of unique values.

        :param embed_dim: int. Embedding dimension.
        """
        super().__init__()
        self.embedding = torch.nn.Embedding(sum(feature_dims), embed_dim)
        self.offsets = np.array((0, *np.cumsum(feature_dims)[:-1]), dtype = np.int64)

    def forward(self, x):
        """
        Perform operations.

        :param x: torch.nn.LongTensor. Contains inputs of size (batch_size, num_features).

        :return: torch.nn.FloatTensor. Contains embeddings of size (batch_size, num_features, 
        embed_dim).
        """
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return self.embedding(x)

class FeaturesLinear(torch.nn.Module):
    def __init__(self, feature_dims, output_dim = 1):
        """
        Linear transformation.

        :param feature_dims: list. List of feature dimension. Each feature contains
        a total number of unique values.

        :param output_dim: int. Embedding dimension.
        """ 

        super().__init__()
        self.linear = torch.nn.Embedding(sum(feature_dims), output_dim)
        self.offsets = np.array((0, *np.cumsum(feature_dims)[:-1]), dtype = np.int64)

        self.bias = torch.nn.Parameter(torch.zeros((output_dim,)))
        
    def forward(self, x):
        """
        Perform operations.

        :param x: torch.nn.LongTensor. Contains inputs of size (batch_size, num_features).

        :return: torch.nn.FloatTensor. Contains linear transformation output of size 
        (batch_size, num_features, embed_dim).
        """
        x = x + x.new_tensor(self.offsets).unsqueeze(0)

        return torch.sum(self.linear(x), dim = 1) + self.bias

class PairwiseInteraction(torch.nn.Module):
    def __init__(self, reduce_sum = True):
        """
        Compute Pairwise Interaction for factorization machine.

        :param reduce_sum: bool. If True, perform reduce sum.
        """
        super().__init__()
        self.reduce_sum = reduce_sum

    def forward(self, x):
        """
         Perform operations.

        :param x: torch.nn.FloatTensor. Contains inputs of size (batch_size, num_features,
        embed_dim).

        :return: torch.nn.FloatTensor. If reduce_sum is True, return pairwise interaction 
        output of size (batch_size, 1), else size (batch_size, embed_dim)
        """
        
        square_of_sum = torch.sum(x, dim=1) ** 2
        sum_of_square = torch.sum(x ** 2, dim=1)
        ix = square_of_sum - sum_of_square
        if self.reduce_sum:
            ix = torch.sum(ix, dim = 1, keepdim = True)
        return 0.5 * ix

class MultilayerPerceptrons(torch.nn.Module):
    def __init__(self, layers, activations, dropouts, batch_normalization = True):
        """
        Multilayer Perceptrons.

        :param layers: list. List of number of nodes which the 1st element refers to 
        the number of nodes for the input layer, while the last element refers to 
        the number of nodes for the output layer. The rest refers to the hidden
        layers.

        :param activations: list. List of activation function for each layer. The number 
        of activation functions = the number of layers - 1.

        :param dropouts: list. List of dropout values. Dropout will not be applied after
        the output layer. The number of dropout values = the number of layers - 2.

        :param batch_normalization: bool. If True, apply batch normalization on each layer 
        (after activation function is applied). Batch normalization will not be 
        applied after the output layer.
        """
        super().__init__()

        mlp_blocks = []
        for i in range(len(layers)-1):
            # Append linear layer
            input_layer = layers[i]
            output_layer = layers[i+1]            
            mlp_blocks.append(torch.nn.Linear(input_layer, output_layer))
            # Append activation function
            activation = activations[i].lower()
            if activation != "linear": 
                mlp_blocks.append(self.activation(activation))
            # Append batch normalization and dropout layers after each layer except output layer
            if batch_normalization:
                if i != len(activations)-1:
                    mlp_blocks.append(torch.nn.BatchNorm1d(output_layer))
            if i != len(activations)-1:
                if dropouts[i] > 0 and dropouts[i] < 1:
                    mlp_blocks.append(torch.nn.Dropout(dropouts[i]))
            
        self.mlp = torch.nn.Sequential(*mlp_blocks)

    def activation(self, name:str = "relu"):
        """
        Convert string values to activation function.

        :param name: str. Name of the activation function. 

        :return: torch.nn. Activation function.
        """
        
        if(name == "relu"): return torch.nn.ReLU()
        elif(name == "relu_true"): return torch.nn.ReLU(True)
        elif(name == "relu6"): return torch.nn.ReLU6()
        elif(name == "relu6_true"): return torch.nn.ReLU6(True)
        elif(name == "elu"): return torch.nn.ELU()
        elif(name == "elu_true"): return torch.nn.ELU(True)
        elif(name == "selu"): return torch.nn.SELU()
        elif(name == "selu_true"): return torch.nn.SELU(True)
        elif(name == "leaky_relu"): return torch.nn.LeakyReLU()
        elif(name == "leaky_relu_true"): return torch.nn.LeakyReLU(True)
        elif(name == "tanh"): return torch.nn.Tanh()
        elif(name == "sigmoid"): return torch.nn.Sigmoid()
        elif(name == "softmax"): return torch.nn.Softmax()
        else: raise ValueError("Unknown non-linearity type")

    def forward(self, x):
        """
        Perform operations.

        :param x: torch.nn.FloatTensor. Contains inputs of size (batch_size, layers[0]).

        :return: torch.nn.FloatTensor. Contains embeddings of size (batch_size, layers[-1]), 
        """

        return self.mlp(x)

def weight_init(m):
    # Weight initialization on specific layer
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
    
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)