import torch

def compute_jacobian(model, inputs):
    """
    Compute the Jacobian of the model outputs with respect to the inputs.
    Args:
        model (nn.Module): The PyTorch model.
        inputs (torch.Tensor): Input tensor with requires_grad=True.
    Returns:
        torch.Tensor: The Jacobian matrix.
    """
    inputs = inputs.clone().detach().requires_grad_(True)
    outputs = model(inputs)
    jacobian = []
    for i in range(outputs.shape[0]):
        grad_outputs = torch.zeros_like(outputs)
        grad_outputs[i] = 1.0
        jac_i = torch.autograd.grad(outputs, inputs, grad_outputs=grad_outputs, retain_graph=True, create_graph=True)[0]
        jacobian.append(jac_i)
    return torch.stack(jacobian)
