import torch

# Done
def weight_init(m, *instances):
    """
    Weight initialization on specific layer.
    """
    if isinstance(m, (instances)):
        torch.nn.init.xavier_uniform_(m.weight)
    
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

# Done
def freeze_partial_linear_params(layer:torch.nn.Linear, 
                                 weight_indices: list, 
                                 bias_indices:list = None, 
                                 dim:int = 0):
    """
    Freeze partial weights on a linear layer.

    - linear (torch.nn.Linear): linear layer.
    - weight_indices (list): Indices which weights to be freezed.
    - bias_indices (list): Indices which biases to be freezed. Default: None.
    - dim (int): Freeze which weight dimension. Default: 0.
    """

    def freezing_hook_weight_full(grad, weight_multiplier):
        return grad * weight_multiplier.to(grad.device)

    def freezing_hook_bias_full(grad, bias_multiplier):
        return grad * bias_multiplier.to(grad.device)

    # Map which weights and biases to be freezed
    weight_multiplier = torch.ones(layer.weight.shape).to(layer.weight.device)
    bias_multiplier = torch.ones(layer.bias.shape).to(layer.bias.device)
    if dim:
        weight_multiplier[:, weight_indices] = 0
    else:
        weight_multiplier[weight_indices] = 0

    if bias_indices:
        bias_multiplier[bias_indices] = 0

    # Register hook
    freezing_hook_weight = lambda grad: freezing_hook_weight_full(grad, weight_multiplier)
    freezing_hook_bias = lambda grad: freezing_hook_bias_full(grad, bias_multiplier)

    weight_hook_handle = layer.weight.register_hook(freezing_hook_weight)
    bias_hook_handle = layer.bias.register_hook(freezing_hook_bias)

    return weight_hook_handle, bias_hook_handle

def l2_regularization(l2_lambda:float, 
                      parameters:dict,
                      device:str = None):
    l2_reg = torch.tensor(0.).to(device)
    for param in parameters:
        l2_reg += torch.norm(param)
    return (l2_reg * l2_lambda)