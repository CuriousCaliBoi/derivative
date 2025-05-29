import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from viz.gradients import plot_heatmap

if __name__ == "__main__":
    results_dir = os.path.join(os.path.dirname(__file__), '../results')
    jacobian = np.load(os.path.join(results_dir, 'jacobian.npy'))
    # Flatten for visualization if needed
    plot_heatmap(jacobian.reshape(-1, jacobian.shape[-1]), title="Jacobian Heatmap") 