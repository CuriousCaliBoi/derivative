import matplotlib.pyplot as plt
import numpy as np

def plot_heatmap(matrix, title="Heatmap", cmap="viridis"):
    """
    Plot a heatmap of a matrix.
    """
    plt.figure(figsize=(6, 5))
    plt.imshow(matrix, aspect='auto', cmap=cmap)
    plt.colorbar()
    plt.title(title)
    plt.xlabel('Columns')
    plt.ylabel('Rows')
    plt.show()

def plot_eigenvalues(matrix, title="Eigenvalue Spectrum"):
    """
    Plot the eigenvalue spectrum of a matrix.
    """
    eigvals = np.linalg.eigvals(matrix)
    plt.figure(figsize=(6, 4))
    plt.plot(np.sort(np.real(eigvals)), 'o')
    plt.title(title)
    plt.xlabel('Index')
    plt.ylabel('Eigenvalue (Real part)')
    plt.show() 