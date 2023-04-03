import numpy as np
from KernelFunction import KernelFunctions

class MyDualPerceptron:

    def __init__(self, kernel_train, kernel_test):
        self.alpha = None #Vettore dei pesi nella forma duale 
        self.b = None #Bias
        self.R = None #Massima distanza euclidea nel mio set di Dati 
        self.K_train = kernel_train #Matrice kernelizzata per poter mappare i dati di train in più dimensioni
        self.K_test = kernel_test   #Matrice kernelizzata per poter mappare i dati di test in più dimensioni

    def train(self, X, y, epochs=1000):

        n_samples, n_features = X.shape
        self.alpha = np.zeros(n_samples)
        self.b = 0
        self.R = np.linalg.norm(X, ord=np.inf)

        for epoch in range(epochs):
            errors = 0
            for i in range(n_samples):
                sum = 0
                for j in range(n_samples):
                    sum += self.alpha[j] * y[j] * self.K_train[i,j]
                y_hat = sum + self.b
                if y[i] * y_hat <= 0:
                    self.alpha[i] += 1
                    self.b += y[i] * self.R ** 2
                    errors += 1
            if errors == 0:
                break

    def predict(self, X, y):
        y_pred = np.zeros(X.shape[0])
        n_samples = len(y_pred)
        for i in range(n_samples):
            sum = 0
            for j in range(n_samples):
                sum += self.alpha[j] * y[j] * self.K_test[i,j]
            y_pred[i] = np.sign(sum + self.b)
        return y_pred