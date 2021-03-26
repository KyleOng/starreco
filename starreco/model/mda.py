import math

import numpy.matlib
import numpy as np
import torch
import torch.nn.functional as F

from starreco.model import (ActivationFunction, 
                            Module)

class mDAEncoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        
        self.weight = torch.nn.Parameter(
            data = torch.Tensor(input_dim, hidden_dim), 
            requires_grad = True
        )
        
        self.weight.data.uniform_(
            -4 * np.sqrt(6. / (hidden_dim + input_dim)), 
            4 * np.sqrt(6. / (hidden_dim + input_dim))
        )
        
        self.bias = torch.nn.Parameter(
            data = torch.ones(hidden_dim), 
            requires_grad = True
        )
        
    def forward(self, x):
        return torch.matmul(x, self.weight) + self.bias
    
class mDADecoder(torch.nn.Module):
    def __init__(self, encoder_weights):
        super().__init__()
        
        self.weight = torch.nn.Parameter(
            data = encoder_weights.T,
            requires_grad = True
        )
        
        self.bias = torch.nn.Parameter(
            data = torch.ones(encoder_weights.shape[0]), 
            requires_grad = True
        )
        
    def forward(self, x):
        return torch.matmul(x, self.weight) + self.bias

class mDA(Module):
    def __init__(self, io_dim:int, hidden_dim:int,
                 noise_rate = 0.3,
                 e_activation = "relu",
                 d_activation = "relu",
                 lr:float = 1e-3,
                 weight_decay:float = 0,
                 criterion:F = None):
        super().__init__(lr, weight_decay, criterion)
        self.noise_rate = noise_rate
        
        encoder = mDAEncoder(io_dim, hidden_dim)
        self.W = encoder.state_dict()["weight"]
        decoder = mDADecoder(self.W)
        self.W_ = decoder.state_dict()["weight"]
        self.encoder = torch.nn.Sequential(
            encoder,    
            ActivationFunction(e_activation)
        )
        self.decoder = torch.nn.Sequential(
            decoder,    
            ActivationFunction(d_activation)
        )
        self.criterion = self.cost_function
    
    def encode(self, x):
        return self.encoder(x)

    def forward(self,x):
        return self.decoder(self.encode(x))
    
    def cost_function(self, y, x):
        """
        Custom cost/objective function. Define in class Module instead of Evaluator, because need to access the weights
        """
        z = self.encode(x)
        
        # Squared loss
        L = torch.mean(torch.sum((x - y) ** 2, axis = 1))
        
        # Regularization term because of Implicit Denoising via Marginalization
        dz = z * (1 - z)
        
        # Reconstruction lost  
        W = self.W.to(self.device)
        W_ = self.W_.to(self.device)
        # 2 * ∑w_^2 * (z(1-z)w)^2  
        # df_x_2 = torch.matmul(dz * dz, torch.matmul(W_ * W_ * 2, W * W))
        df_x_2 = torch.matmul(torch.matmul(dz * dz, W_ * W_ * 2), W * W)
        L2 = self.noise_rate * self.noise_rate * torch.mean(torch.sum(df_x_2, axis = 1))
        cost = L + 0.5 * L2
        
        return cost

def mapping(x:torch.FloatTensor, p:float, bias = True):
    d = x.shape[-1]
    q = torch.ones((d, 1)) * (1 - p)
    # Don't corrupt the bias term
    q[-1] = 1 if bias else q[-1]
    S = torch.matmul(x.T, x)
    Q = S * torch.matmul(q, q.T)
    v = q[:,0] * S.diag()
    mask = torch.diag(torch.ones_like(v))
    Q = mask * torch.diag(v) + (1 - mask) * Q
    P = S * torch.tile(q.T, (d, 1))
    A = Q + 1e-5 * torch.eye(d)
    B = P[:d-1, :] if bias else P
    W = torch.linalg.solve(A.T, B.T)
    return W

def LmDA(x:torch.FloatTensor, p:float, return_data = "non_linear", bias = True):
    assert return_data in ["linear", "non_linear", "mapping"], \
    "`return_data` can be linear, non_linear or mapping only."

    # Add bias
    if bias:
        x = torch.hstack((x, torch.ones(x.shape[0], 1)))
        W = mapping(x, p)
    else:
        W = mapping(x, p, False)
    x = torch.matmul(x, W)
    if return_data == "linear":
        return x
    elif return_data == "non_linear":
        return torch.selu(x)
    elif return_data == "mapping":
        return W