import numpy as np
from KernelFunction import KernelFunctions

class MyDualPerceptron:

    def __init__(self):
        self.a = None #Vettore dei pesi nella forma duale 
        self.b = None #Bias
        self.R = None #Massima distanza euclidea nel mio set di Dati 

    def train(self, X, y, kernel, epochs=1000):
 
        n_samples = X.shape[0]
        self.a = np.zeros(n_samples)
        self.b = 0
        self.R = np.max(np.linalg.norm(X, axis=1))
        
        for k in range(epochs):
            n_err = 0
            for i in range(n_samples):
                prediction = sum(self.a[j] * y[j] * kernel(X[j],X[i]) + self.b for j in range(n_samples))
                if y[i] * prediction <= 0:
                    self.a[i] += 1
                    self.b += y[i] * self.R**2
                    n_err += 1
            if n_err == 0:
                break
            
        return self.a, self.b
    
    """
    def predict(self, X, y):

        #Inizializzo 
        n_samples = X.shape[0]
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
                sum += self.a[j] * y[j] * K_test[i,j]
            y_pred[i] = np.sign(sum + self.b)

        return y_pred
    """

    def predict(self, X_test, X_train, y_train, a, b, kernel):
        y_pred = []
        for x_test in X_test:
            prediction = sum(a[j] * y_train[j] * kernel(X_train[j], x_test) for j in range(len(X_train))) + b
            y_pred.append(np.sign(prediction))
        return np.array(y_pred)