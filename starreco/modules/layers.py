from typing import Union
import math

import numpy as np
import torch

# Done
class FeaturesEmbedding(torch.nn.Module):
    """
    Feature Embedding class.

    - field_dims (list): List of field dimension. Each feature contains a total number of unique values.
    - embed_dim (int): Embedding dimension.
    """

    def __init__(self, 
                 field_dims: list, 
                 embed_dim: int):
        super().__init__()
        self.embedding = torch.nn.Embedding(sum(field_dims), embed_dim)
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype = np.int64)

    def forward(self, x):
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return self.embedding(x)

# Done
class FeaturesLinear(torch.nn.Module):
    """
    Linear transformation class.

    - field_dims (list): List of field dimension. Each feature contains a total number of unique values.
    - output_dim (int): Output dimension. Default 1.
    """ 

    def __init__(self, 
                 field_dims:list, 
                 output_dim:int = 1):
        super().__init__()
        self.linear = torch.nn.Embedding(sum(field_dims), output_dim)
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype = np.int64)

        self.bias = torch.nn.Parameter(torch.zeros((output_dim,)))
        
    def forward(self, x):
        x = x + x.new_tensor(self.offsets).unsqueeze(0)

        return torch.sum(self.linear(x), dim = 1) + self.bias

# Done
class PairwiseInteraction(torch.nn.Module):
    """
    Pairwise interaction class for factorization machine.

    - reduce_sum (bool): If True, perform reduce sum. Default: True.
    """

    def __init__(self, reduce_sum:bool = True):
        super().__init__()
        self.reduce_sum = reduce_sum

    def forward(self, x):        
        square_of_sum = torch.sum(x, dim=1) ** 2
        sum_of_square = torch.sum(x ** 2, dim=1)
        ix = square_of_sum - sum_of_square
        if self.reduce_sum:
            ix = torch.sum(ix, dim = 1, keepdim = True)
        return 0.5 * ix

# Done
class ActivationFunction(torch.nn.Module):
    """
    Activation Function class which convert string to activation function layer.

    - name (str): Name of the activation function. Default: "relu".
    """

    def __init__(self, name:str = "relu"):
        super().__init__()
        name = name.lower()
        
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
        elif (name == "linear"): self.activation = None
        else: raise ValueError("Unknown non-linearity type")

    def forward(self, x):
        return self.activation(x) if self.activation else x

# Done
class MultilayerPerceptrons(torch.nn.Module):
    """
    Multilayer Perceptrons with dynamic settings.

    - input_dim (int): Number of neurons in the input layer. 
    - hidden_dims (list): List of numbers of neurons throughout the hidden layers. 
    - activations (str/list): List of activation functions. Default: "relu".
    - dropouts (int/float/list): List of dropout values. Default: 0.5.
    - batch_norm (bool): If True, apply batch normalization in every layer. Batch normalization is applied between activation and dropout layer. Default: True.
    - remove_last_dropout (bool): If True, remove batch normalization at the last hidden layer. Do this if the last hidden layer is the output layer. Default: False.
    - remove_last_batch_norm (bool): If True, remove dropout at the last hidden layer. Do this if the last hidden layer is the output layer. Default: False.
    - output_layer (str): Activation applied to the output layer which only output 1 neuron. Set as None, if your want your last hidden layer to be the output layer. Default: "linear".
    - extra_input_dims (int/list): List of extra input dimension at every layer. Default: 0.
    - extra_output_dims (int/list): List of extra output dimension at every layer. Extra output dimension is not applied to the output layer, if `output_layer` is set with value. Default: 0.
    - mlp_type (str): Return MLP type. Default: "sequential".
    """

    def __init__(self, 
                 input_dim:int, 
                 hidden_dims:list = [], 
                 activations:Union[str, list] = "relu",
                 dropouts:Union[int, float, list] = 0.5,
                 batch_norm:bool = True,
                 remove_last_dropout:bool = False,
                 remove_last_batch_norm:bool = False,
                 output_layer:str = "linear",
                 extra_input_dims:Union[int,list] = 0,
                 extra_output_dims:Union[int,list] = 0,
                 mlp_type:str = "sequential"):
        super().__init__()

        mlp_blocks = []
        if hidden_dims:
            # Transform float-to-list arguments to list
            if type(activations) == str:
                activations = [activations] * len(hidden_dims)
            if type(dropouts) == float or type(dropouts) == int:
                dropouts = [dropouts] * len(hidden_dims)
            if type(extra_input_dims) == int:
                extra_input_dims = [extra_input_dims] * len(hidden_dims)
            if type(extra_output_dims) == int:
                extra_output_dims = [extra_output_dims] * len(hidden_dims)

            # Error messages for any violations
            assert len(extra_input_dims) == len(hidden_dims), "`hidden_dims` and `extra_input_dims` must have equal length."
            assert len(extra_output_dims) == len(hidden_dims), "`hidden_dims` and `extra_output_dims` must have equal length."
            assert max(dropouts) >= 0 and max(dropouts) <= 1, "maximum `dropouts` must in between 0 and 1,"
            assert mlp_type in ["sequential", "modulelist"], "`mlp_type` must be 'sequential' or 'modulelist'"

            for i in range(len(hidden_dims)):
                output_dim = hidden_dims[i]

                # Linear layer         
                mlp_blocks.append(torch.nn.Linear(input_dim + extra_input_dims[i], output_dim + extra_output_dims[i]))

                # Batch normalization
                # Batch normalization will not be applied to the last hidden layer (output layer is None)
                if batch_norm:
                    if i + int(remove_last_batch_norm) != len(hidden_dims):
                        mlp_blocks.append(torch.nn.BatchNorm1d(output_dim))

                # Activation function
                activation = activations[i].lower()
                mlp_blocks.append(ActivationFunction(activation))

                # Dropout
                # Dropout will not be applied to the last hidden layer (output layer is None)
                if dropouts[i] > 0 and dropouts[i] <= 1:
                    if i + int(remove_last_dropout) != len(hidden_dims):
                        mlp_blocks.append(torch.nn.Dropout(dropouts[i]))
                
                # Replace input dim with current output_dim
                input_dim = output_dim
            
        if output_layer:
            mlp_blocks.append(torch.nn.Linear(input_dim, 1))
            mlp_blocks.append(ActivationFunction(output_layer))
            
        if mlp_type == "sequential":
            self.mlp = torch.nn.Sequential(*mlp_blocks) 
        elif mlp_type == "modulelist":
            self.mlp = torch.nn.ModuleList(mlp_blocks)

    def forward(self, x):
        return self.mlp(x)

# Done
class StackedDenoisingAutoEncoder(torch.nn.Module):
    """
    Stacked Denoising AutoEncoder class with dynamic setting.

    - input_output_dim (int): Number of neurons in the input and output layer.
    - hidden_dims (list): List of number of neurons throughout the encoder and decoder (reverse list) hidden layers. Default: [512, 256, 128].
    - e_activations (str/list): List of activation functions in the encoder layers. Default: "relu".
    - d_activations (str/list): List of activation functions in the decoder layers. Default: "relu".
    - e_dropouts (int/float/list): List of dropout values in the encoder layers. Default: 0.
    - d_dropouts (int/float/list): List of dropout values in the decoder layers. Default: 0.
    - dropout (float): Dropout value in the latent space. Default: 0.5.
    - batch_norm (bool): If True, apply batch normalization in all hidden layers. Default: True.
    - extra_input_dims (int/list): List of extra input dimension at every layer. Default: 0.
    - extra_output_dims (int/list): List of extra output dimension at every layer. Extra output dimension is not applied to the output layer, if `output_layer` is set with value. Default: 0.
    - noise_rate (int/float): Rate/Percentage of noises to be added to the input. Noise is not applied to extra input neurons. Noise is only applied during training only. Default: 1.
    - noise_factor (int/float): Noise factor. Default: 1.
    - noise_all (bool): If True, noise are added to inputs in all input and hidden layers, else only to input layer. Default: False.
    - mean (int/float): Gaussian noise mean. Default: 0.
    - std (int/float): Gaussian noise standard deviation: 1.
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
                 extra_input_dims:Union[int,list] = 0,
                 extra_output_dims:Union[int,list] = 0,
                 noise_rate:Union[int, float] = 1,
                 noise_factor:Union[int, float] = 1,
                 noise_all:bool = True,
                 noise_mask:bool = False,
                 mean:Union[int, float] = 0,
                 std:Union[int, float] = 1):
        super().__init__()

        self.noise_rate = noise_rate
        self.noise_factor = noise_factor
        self.noise_all = noise_all
        self.mean = mean
        self.std = std

        # if extra dim is type int, transform them to list
        if type(extra_input_dims) == int:
            extra_input_dims = [extra_input_dims] * (len(hidden_dims) * 2)
        if type(extra_output_dims) == int:
            extra_output_dims = [extra_output_dims] * (len(hidden_dims) * 2)
        
        # Error messages for any violations
        assert len(extra_input_dims) == len(hidden_dims) * 2, ""
        assert len(extra_output_dims) == len(hidden_dims) * 2, ""

        encoder_extra_input_dims = extra_input_dims[:len(extra_input_dims)//2]
        encoder_extra_output_dims = extra_output_dims[:len(extra_output_dims)//2]
        decoder_extra_input_dims = extra_input_dims[len(extra_input_dims)//2:]
        decoder_extra_output_dims = extra_output_dims[len(extra_output_dims)//2:]

        # Encoder layer
        self.encoder = MultilayerPerceptrons(input_dim = input_output_dim,
                                             hidden_dims = hidden_dims, 
                                             activations = e_activations, 
                                             dropouts = e_dropouts,
                                             batch_norm = batch_norm,
                                             remove_last_dropout = True,
                                             remove_last_batch_norm = False,
                                             output_layer = None,
                                             extra_input_dims = encoder_extra_input_dims,
                                             extra_output_dims = encoder_extra_output_dims,
                                             mlp_type = "modulelist")

        # Decoder layer 
        # Set the decoder last activation to ReLU
        if len(hidden_dims) > 1:
            d_activations = [d_activations] * (len(hidden_dims) - 1) + ["relu"]
        else:
            d_activations = ["relu"]
        self.decoder = MultilayerPerceptrons(input_dim = hidden_dims[-1],
                                             hidden_dims = [*hidden_dims[:-1][::-1], input_output_dim], 
                                             activations = d_activations, 
                                             dropouts = d_dropouts,
                                             batch_norm = batch_norm,
                                             remove_last_dropout = True,
                                             remove_last_batch_norm = True,
                                             output_layer = None,
                                             extra_input_dims = decoder_extra_input_dims,
                                             extra_output_dims = decoder_extra_output_dims,
                                             mlp_type = "modulelist")
        
        # Batch normalization and dropout layer inserted before decoding
        if batch_norm:
            self.decoder.mlp.insert(0, torch.nn.BatchNorm1d(hidden_dims[-1]))
        if dropout:
            self.decoder.mlp.insert(1, torch.nn.Dropout(dropout))

    def add_noise(self, x):
        # Get noise mask
        ones = torch.ones(math.ceil(self.noise_rate * x.nelement()))
        zeros = torch.zeros(x.nelement() - ones.nelement())
        noise_mask = torch.cat([zeros, ones])
        noise_mask = noise_mask[torch.randperm(noise_mask.shape[0])]
        noise_mask = noise_mask.view(x.shape).to(x.device).bool()

        noise = torch.randn(x.shape).to(x.device) * self.std + self.mean
        noise *= noise_mask
        noise *= self.noise_factor

        return x + noise, noise_mask

    def encode(self, x, extra = None, add_noise = True):
        # Reset noise_masks array during training and if add_noise boolean is True
        if self.training and add_noise:
            self.encoder_noise_masks = []

        for i, module in enumerate(self.encoder.mlp):
            if type(module) == torch.nn.Linear:
                # Noise add only during training
                if not(not self.noise_all and i) and self.training and add_noise:
                    x, noise_mask = self.add_noise(x)
                    # Append noise mask
                    self.encoder_noise_masks.append(noise_mask)
                # Concatenate extra to x
                if extra is not None and module.in_features > x.shape[-1]:
                    x = torch.cat([x, extra], dim = 1)
            x = module(x)
        return x

    def decode(self, x, extra = None, add_noise = True):
        # Reset noise_masks array during training and if add_noise boolean is True
        if self.training and add_noise:
            self.decoder_noise_masks = []

        for i, module in enumerate(self.decoder.mlp):
            if type(module) == torch.nn.Linear:
                # Noise add only during training
                if self.noise_all and self.training and add_noise:
                    x, noise_mask = self.add_noise(x)
                    # Append noise mask
                    self.decoder_noise_masks.append(noise_mask)
                # Concatenate extra to x
                if extra is not None and module.in_features > x.shape[-1]:
                    x = torch.cat([x, extra], dim = 1)
            x = module(x)
        return x

    def forward(self, x, extra = None):
        x = self.decode(self.encode(x, extra), extra)
        return x       

# Need testing
class CompressedInteraction(torch.nn.Module):
    """
    Compressed Interaction class.

    - input_dim (int): Number of neurons in the input layer. 
    - cross_dims (list): List of number of neuron for cross dimensions.
    - activations (str/list): List of activation functions. Default: "relu".
    - split_half (bool): If True, convolution output is splitted into half of the 1st dimension. Default: True.
    """

    def __init__(self, 
                 input_dim:int, 
                 cross_dims:list, 
                 activation:str = "relu", 
                 split_half:bool = True):
        super().__init__()
        self.num_layers = len(cross_dims)
        self.split_half = split_half

        convolution_blocks = []
        prev_dim, fc_input_dim = input_dim, 0
        for i in range(self.num_layers):
            input_channel_size = input_dim * prev_dim
            output_channel_size = cross_dims[i]
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

# Need testing
class InnerProduct(torch.nn.Module):
    """
    Inner Product class.

    - reduce_sum (bool): If True, perform reduce sum. Default: True.
    """

    def __init__(self, reduce_sum:bool = True):
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