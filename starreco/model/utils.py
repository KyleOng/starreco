import torch

# Done
def weight_init(m):
    """Weight initialization on specific layer."""
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
    
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

# Done
def freeze_partial_linear_params(layer, weight_indices, bias_indices = None, dim = 0):
    """Freeze partial weights on a linear layer."""
    def freezing_hook_weight_full(grad, weight_multiplier):
        return grad * weight_multiplier.to(grad.device)

    def freezing_hook_bias_full(grad, bias_multiplier):
        return grad * bias_multiplier.to(grad.device)

    # Map which weights to be freezed
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