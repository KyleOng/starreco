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

        :param x: torch.LongTensor. Contains inputs of size (batch_size, num_features).

        :return: torch.FloatTensor. Contains embeddings of size (batch_size, num_features, 
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

        :param x: torch.LongTensor. Contains inputs of size (batch_size, num_features).

        :return: torch.FloatTensor. Contains linear transformation output of size 
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

        :param x: torch.FloatTensor. Contains inputs of size (batch_size, num_features,
        embed_dim).

        :return: torch.FloatTensor. If reduce_sum is True, return pairwise interaction 
        output of size (batch_size, 1), else size (batch_size, embed_dim)
        """
        
        square_of_sum = torch.sum(x, dim=1) ** 2
        sum_of_square = torch.sum(x ** 2, dim=1)
        ix = square_of_sum - sum_of_square
        if self.reduce_sum:
            ix = torch.sum(ix, dim = 1, keepdim = True)
        return 0.5 * ix

class ActivationFunction(torch.nn.Module):
    def __init__(self, name:str = "relu"):
        """
        Convert string values to activation function.

        :param name: str. Name of the activation function. 
        """
        super().__init__()
        
        if (name == "relu"): self.activation = torch.nn.ReLU()
        elif (name == "relu_true"): self.activation = torch.nn.ReLU(True)
        elif (name == "relu6"): self.activation = torch.nn.ReLU6()
        elif (name == "relu6_true"): self.activation = torch.nn.ReLU6(True)
        elif (name == "elu"): self.activation = torch.nn.ELU()
        elif (name == "elu_true"): self.activation = torch.nn.ELU(True)
        elif (name == "selu"): self.activation = torch.nn.SELU()
        elif (name == "selu_true"): self.activation = torch.nn.SELU(True)
        elif (name == "leaky_relu"): self.activation = torch.nn.LeakyReLU()
        elif (name == "leaky_relu_true"): self.activation = torch.nn.LeakyReLU(True)
        elif (name == "tanh"): self.activation = torch.nn.Tanh()
        elif (name == "sigmoid"): self.activation = torch.nn.Sigmoid()
        elif (name == "softmax"): self.activation = torch.nn.Softmax()
        else: raise ValueError("Unknown non-linearity type")

    def forward(self, x):
        """
        Perform non linear activation function on input

        :param x: torch.nn.Tensor. Input of any sizes.

        :return: torhc.nn.Tensor. Non linear output.
        """
        return self.activation(x)

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
                mlp_blocks.append(ActivationFunction(activation))
            # Append batch normalization and dropout layers after each layer except output layer
            if batch_normalization:
                if i != len(activations)-1:
                    mlp_blocks.append(torch.nn.BatchNorm1d(output_layer))
            if i != len(activations)-1:
                if dropouts[i] > 0 and dropouts[i] < 1:
                    mlp_blocks.append(torch.nn.Dropout(dropouts[i]))
            
        self.mlp = torch.nn.Sequential(*mlp_blocks)

    def forward(self, x):
        """
        Perform operations.

        :param x: torch.FloatTensor. Contains inputs of size (batch_size, layers[0]).

        :return: torch.FloatTensor. Contains embeddings of size (batch_size, layers[-1]), 
        """

        return self.mlp(x)

class CompressedInteraction(torch.nn.Module):
    def __init__(self, input_dim, cross_layers, activation = "relu", split_half = True):
        super().__init__()
        self.num_layers = len(cross_layers)
        self.split_half = split_half

        convolution_blocks = []
        prev_dim, fc_input_dim = input_dim, 0
        for i in range(self.num_layers):
            input_channel_size = input_dim * prev_dim
            output_channel_size = cross_layers[i]
            convolution_blocks.append(torch.nn.Conv1d(input_channel_size, output_channel_size, 1,
            stride = 1, dilation = 1, bias = True))
            if self.split_half and i != self.num_layers - 1:
                output_channel_size //= 2
            prev_dim = output_channel_size
            fc_input_dim += prev_dim
        convolution_blocks.append(ActivationFunction(activation))
        self.convolution = torch.nn.Sequential(*convolution_blocks)

        self.fc = torch.nn.Linear(fc_input_dim, 1)
        
    def forward(self, x):
        xs = list()
        x0, h = x.unsqueeze(2), x
        for i in range(self.num_layers):
            x = x0 * h.unsqueeze(1)
            batch_size, f0_dim, fin_dim, embed_dim = x.shape
            x = x.view(batch_size, f0_dim * fin_dim, embed_dim)
            x = self.convolution[i](x)
            if self.split_half and i != self.num_layers - 1:
                x, h = torch.split(x, x.shape[1] // 2, dim=1)
            else:
                h = x
            xs.append(x)
        return self.fc(torch.sum(torch.cat(xs, dim=1), 2))

class InnerProduct(torch.nn.Module):
    def __init__(self, reduce_sum = True):
        super().__init__()
        self.reduce_sum = reduce_sum

    def forward(self, x):
        embed_list = x
        row = []
        col = []
        num_inputs = len(embed_list)

        for i in range(num_inputs - 1):
            for j in range(i + 1, num_inputs):
                row.append(i)
                col.append(j)

        p = torch.cat([embed_list[idx] for idx in row], dim=1)
        q = torch.cat([embed_list[idx] for idx in col], dim=1)

        inner_product = p * q
        if self.reduce_sum:
            inner_product = torch.sum(inner_product, dim = 2, keepdim = True)
        return inner_product

def weight_init(m):
    # Weight initialization on specific layer
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
    
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)