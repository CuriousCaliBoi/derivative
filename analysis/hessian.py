import torch

def compute_hessian(model, inputs):
    """
    Compute the Hessian of a scalar model output with respect to the inputs.
    Args:
        model (nn.Module): The PyTorch model.
        inputs (torch.Tensor): Input tensor with requires_grad=True.
    Returns:
        torch.Tensor: The Hessian matrix.
    """
    inputs = inputs.clone().detach().requires_grad_(True)
    output = model(inputs)
    if output.numel() != 1:
        raise ValueError("Model output must be a scalar for Hessian computation.")
    hessian = torch.autograd.functional.hessian(lambda x: model(x), inputs)
    return hessian 