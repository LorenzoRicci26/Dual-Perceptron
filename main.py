import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from DualPerceptron import DualPerceptron
from KernelFunction import KernelFunctions


def main():
    df = pd.read_csv('cmp\iris_data.csv', names = ["Sepal Lenght", "Sepal Width", "Petal Lenght", "Petal Width", "Class"])
    
    X = df.iloc[: , :4].values
    y = df.iloc[:, 4].map({'Iris-setosa' : -1, 'Iris-versicolor' : 1, 'Iris-virginica' : 1}).values

    kernel_function = KernelFunctions()
    kernel_matrix = None

    print('Quale tipologia di Kernel vuoi utilizzare? 1: polynomial_kernel, 2: gaussian_kernel, 3: rbf_kernel')

    kernel_type = input('Choose : ')

    if kernel_type == '1':
        kernel_matrix = kernel_function.polynomial_kernel(X, X, degree=3)
    elif kernel_type == '2':
        kernel_matrix = kernel_function.gaussian_kernel(X, X, sigma = 0.5)
    elif kernel_type == '3':
        kernel_matrix = kernel_function.rbf_kernel(X, X, gamma = 0.3)
    else:
        print("Hai inserito un valore di input errato !!!")

    """
    # Crea una figura e un'area degli assi
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Traccia gli esempi di Iris Setosa con cerchi rossi
    ax.scatter(X[y == -1, 0], X[y == -1, 1], X[y ==-1, 2], X[y==-1, 3], c='r', marker='o', label='Iris Setosa')

    # Traccia gli esempi di Iris Versicolor e Virginica con cerchi blu
    ax.scatter(X[y == 1, 0], X[y == 1, 1], X[y == 1, 2], X[y== 1, 3], c='b', marker='o', label='Iris Versicolor and Virginica')

    # Aggiungi una legenda
    ax.legend()

    # Aggiungi etichette degli assi
    ax.set_xlabel('Sepal Length')
    ax.set_ylabel('Sepal Width')
    ax.set_zlabel('Petal Lenght')

    # Mostra la figura
    plt.show()
    """
    
    perceptron = DualPerceptron(kernel_matrix)

    perceptron.train(X, y)

    y_predicted = perceptron.predict(X, y)

    print(y_predicted)
    
       
if __name__ == "__main__":
    main()