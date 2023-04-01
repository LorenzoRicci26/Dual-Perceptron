import numpy as np

class KernelFunctions:

    def __init__(self) -> None:
        pass
        

    def rbf_kernel(self, x1, x2, gamma = 0.5):
            
        euclidean_distance = np.linalg.norm(x2 - x1)
        output = np.exp(-(euclidean_distance / 2 * (gamma ** 2)))
        return output

    """
    def rbf_kernel(self, X, var, gamma):
        X_norm = np.sum(X ** 2, axis = -1)
        K = var * np.exp(-gamma * (X_norm[:,None] + X_norm[None,:] - 2 * np.dot(X, X.T)))
        return K
    """
    def gaussian_kernel(self, l=5, sigma=1.):
    
        #creates gaussian kernel with side length `l` and a sigma of `sigma`

        ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
        gauss = np.exp(-0.5 * np.square(ax) / np.square(sigma))
        kernel = np.outer(gauss, gauss)
        K = kernel / np.sum(kernel)
        return K
    

    def polynomial_kernel(self, x1, x2, dimension=2):
        return (np.dot(x1, x2) + 1) ** dimension
