import torch

def masked_mse_loss(y_hat, y, masked_value = 0):
    """
    Masked mean square error loss.
    """
    mask = y != masked_value
    masked_se = torch.square(mask * (y - y_hat))
    masked_mse = torch.sum(masked_se) / torch.sum(mask)
    return masked_mse