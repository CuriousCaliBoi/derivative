import sys

import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from models.transformer import MinimalTransformerEncoder
from analysis.jacobian import compute_jacobian
from analysis.hessian import compute_hessian
import numpy as np


def main():
    """
    Run Jacobian and Hessian analysis experiments.
    """
    model = MinimalTransformerEncoder()
    x = torch.randn(1, 10, 32, requires_grad=True)
    jacobian = compute_jacobian(model, x)
    hessian = compute_hessian(model, x)
    results_dir = os.path.join(os.path.dirname(__file__), '../results')
    os.makedirs(results_dir, exist_ok=True)
    np.save(os.path.join(results_dir, "jacobian.npy"), jacobian.detach().numpy())
    np.save(os.path.join(results_dir, "hessian.npy"), hessian.detach().numpy())
    print(f"Analysis complete. Results saved to {results_dir}")

if __name__ == "__main__":
    main()
