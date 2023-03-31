import numpy as np

class DualPerceptron:

    def __init__(self):
        self.alpha = None
        self.b = None
        self.R = None

    def train(self, X, y, epochs=1000):
        n_samples, n_features = X.shape
        self.alpha = np.zeros(n_samples)
        self.b = 0
        self.R = np.linalg.norm(X, ord=np.inf)

        for epoch in range(epochs):
            errors = 0
            for i in range(n_samples):
                y_hat = self.summatory(i, self.alpha.shape[0], X, y) + self.b
                if y[i] * y_hat <= 0:
                    self.alpha[i] += 1
                    self.b += y[i] * self.R ** 2
                    errors += 1
            if errors == 0:
                break
    
    def summatory(self, i, l, X, y):
        sum = 0
        for j in range (l):
            scalar_product = np.dot(X[j], X[i])
            sum += self.alpha[j] * y[j] * scalar_product
        return sum
    
    def predict(self, X, y):
        y_pred = np.zeros(X.shape[0])
        for i in range(y_pred.shape[0]):
            y_pred[i] = self.decision_function(i, y.shape[0], X, y)
        return y_pred
    
    def decision_function(self, i, l, X, y):
        return np.sign(self.summatory(i, l, X, y) + self.b)


    """
    def __init__(self, kernel_func, max_iterations=1000):
        self.kernel_func = kernel_func
        self.max_iterations = max_iterations

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Inizializzazione dei pesi alpha e del bias b
        self.alpha = np.zeros(n_samples)
        self.b = 0

        # Calcolo della matrice del kernel
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i,j] = self.kernel_func(X[i], X[j])

        # Iterazioni dell'algoritmo
        for _ in range(self.max_iterations):
            mistakes = 0
            for i in range(n_samples):
                # Calcolo dell'output del classificatore per l'i-esimo esempio
                output_i = np.sum(self.alpha * y * K[:,i]) + self.b

                # Aggiornamento dei pesi alpha e del bias b in caso di errore
                if np.sign(output_i) != y[i]:
                    self.alpha[i] += 1
                    self.b += y[i]
                    mistakes += 1

            # Se non ci sono stati errori in questa iterazione, termina
            if mistakes == 0:
                break

    def predict(self, X):
        n_samples = X.shape[0]

        # Calcolo dei valori di output per tutti gli esempi di test
        y_pred = np.zeros(n_samples)
        for i in range(n_samples):
            output_i = np.sum(self.alpha * self.y * self.kernel_func(X[i], self.X)) + self.b
            y_pred[i] = np.sign(output_i)

        return y_pred
"""