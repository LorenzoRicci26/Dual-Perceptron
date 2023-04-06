import numpy as np
from KernelFunction import KernelFunctions

class MyDualPerceptron:

    def __init__(self, kernel_type):
        self.alpha = None #Vettore dei pesi nella forma duale 
        self.b = None #Bias
        self.R = None #Massima distanza euclidea nel mio set di Dati 
        self.kernel = KernelFunctions() 
        self.kernel_type = kernel_type

    def train(self, X, y, epochs=1000):
 
        n_samples, n_features = X.shape
        self.alpha = np.zeros(n_samples)
        self.b = 0
        self.R = np.linalg.norm(X, ord=np.inf)
        K_train = np.zeros((n_samples,n_samples))
        
        #Inserisco all'interno della matrice kernelizzata per il train i valori a seconda del tipo di kernel scelto
        if self.kernel_type == '1':
            K_train = self.kernel.linear_kernel(X)
        elif self.kernel_type == '2':
            K_train = self.kernel.polynomial_kernel(X)
        elif self.kernel_type == '3':
            K_train = self.kernel.rbf_kernel(X)
        else:
            print('Hai inserito un input non corretto !!!')
            return

        for epoch in range(epochs):
            errors = 0
            for i in range(n_samples):
                sum = 0
                for j in range(n_samples):
                    sum += self.alpha[j] * y[j] * K_train[i,j]
                y_hat = sum + self.b
                if y[i] * y_hat <= 0:
                    self.alpha[i] += 1
                    self.b += y[i] * self.R ** 2
                    errors += 1
            if errors == 0:
                break
        
        print('Il vettore dei pesi in forma duale è: ' + str(self.alpha))
        print('Il bias è: ' + str(self.b))

    def predict(self, X, y):

        #Inizializzo 
        n_samples, n_features = X.shape
        y_pred = np.zeros(n_samples)
        K_test = np.zeros((n_samples,n_samples))

        #Inserisco all'interno della matrice kernelizzata per il test i valori a seconda del tipo di kernel scelto
        if self.kernel_type == '1':
            K_test = self.kernel.linear_kernel(X)
        elif self.kernel_type == '2':
            K_test = self.kernel.polynomial_kernel(X)
        elif self.kernel_type == '3':
            K_test = self.kernel.rbf_kernel(X)
        else:
            print('Hai inserito un input non corretto !!!')
            return
        
        #Ciclo che mi calcola il vettore predetto y_pred secondo la funzione di associazione np.sign(sum + self.b)
        for i in range(n_samples):
            sum = 0
            for j in range(n_samples):
                sum += self.alpha[j] * y[j] * K_test[i,j]
            y_pred[i] = np.sign(sum + self.b)

        return y_pred