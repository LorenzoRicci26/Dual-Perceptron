import numpy as np

class KernelFunctions:

    def __init__(self) -> None:
        pass

    def linear_kernel(self, x1, x2):
        return np.dot(x1, x2)

    def polynomial_kernel(self, x1, x2, degree=2):
        return (np.dot(x1, x2) + 1)**degree

    def rbf_kernel(self, x1, x2, gamma=0.15):
        distance = np.linalg.norm(x1 - x2) ** 2
        return np.exp(-gamma * distance)
    