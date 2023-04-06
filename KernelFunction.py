import numpy as np

class KernelFunctions:

    def __init__(self) -> None:
        pass

    def rbf_kernel(self, X):

        """
        RBF kernel mapping for a given set of data X.
        Parameters:
            X: array-like, shape = [n_samples, n_features]
                Input data.
            gamma: float, default = 1
                Hyperparameter for tuning the influence of each sample.
        Returns:
            K: array-like, shape = [n_samples, n_samples]
                Kernel matrix of the mapped data.
        """

        n_samples, n_features = X.shape
        K = np.zeros((n_samples, n_samples))
        gamma = 1000
        for i in range(n_samples):
            for j in range(n_samples):
                K[i,j] = np.exp(-(gamma * np.linalg.norm(X[i] - X[j])**2))
        return K

    def polynomial_kernel(self, X, c = 1, d = 3):
        return (np.dot(X, X.T) + c ) ** d

    def linear_kernel(self, X):
        return np.dot(X, X.T)
    