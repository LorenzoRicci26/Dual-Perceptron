import numpy as np

class KernelFunctions:

    def __init__(self) -> None:
        pass
        
    
    def rbf_kernel(self, X, var, gamma):
        X_norm = np.sum(X ** 2, axis = -1)
        K = var * np.exp(-gamma * (X_norm[:,None] + X_norm[None,:] - 2 * np.dot(X, X.T)))
        return K


