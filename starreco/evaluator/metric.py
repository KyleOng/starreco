from typing import Union

import torch

def mDA_reconstruction_loss(x:torch.tensor, 
                            y:torch.tensor, 
                            z:torch.tensor, 
                            W:torch.tensor, 
                            W_:torch.tensor, 
                            noise_rate:Union[float, int], 
                            device:torch.device):
    """
    Reconstruction loss functions for marginalized Denoising Autoencoder.
      
    x (torch.tensor): Input x.

    y (torch.tensor): Output prediction y.

    z (torch.tensor): Encoded/Compressed x.

    W (torch.tensor): mDA Encoder weights.

    W_ (torch.tensor): mDA Decoder weights.

    noise_rate (int/float): noise rate.

    device (torch.device): Device which W and W_ will be allocated.
    """
    # Move weights to specified device.
    W = W.to(device)
    W_ = W_.to(device)

    # Squared loss
    L = torch.mean(torch.sum((x - y) ** 2, axis = 1))
    
    # Regularization term because of Implicit Denoising via Marginalization
    dz = z * (1 - z)
    
    # Reconstruction lost  
    # 2 * ∑w_^2 * (z(1-z)w)^2  
    # df_x_2 = torch.matmul(dz * dz, torch.matmul(W_ * W_ * 2, W * W))
    df_x_2 = torch.matmul(torch.matmul(dz * dz, W_ * W_ * 2), W * W)
    L2 = noise_rate * noise_rate * torch.mean(torch.sum(df_x_2, axis = 1))
    cost = L + 0.5 * L2
    
    return cost
