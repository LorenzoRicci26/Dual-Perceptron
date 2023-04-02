import numpy as np

class KernelFunctions:

    def __init__(self) -> None:
        pass

    def rbf_kernel(self, X, Y, gamma):

        # Calcola le distanze quadrate tra le righe di X e Y
        dist_squared = np.sum(X**2, axis=1)[:, np.newaxis] + np.sum(Y**2, axis=1) - 2 * np.dot(X, Y.T)

        # Calcola il kernel RBF
        return np.exp(-gamma * dist_squared)

    def polynomial_kernel(self, X, Y, degree=2):

        # Calcola il prodotto tra le righe di X e Y
        dot_product = np.dot(X, Y.T)

        # Calcola il kernel polinomiale
        return (dot_product + 1) ** degree

    def gaussian_kernel(self, X, Y, sigma):

        # Calcola le distanze quadrate tra le righe di X e Y
        dist_squared = np.sum(X**2, axis=1)[:, np.newaxis] + np.sum(Y**2, axis=1) - 2 * np.dot(X, Y.T)

        # Calcola il kernel gaussiano
        return np.exp(-dist_squared / (2 * sigma**2))
    