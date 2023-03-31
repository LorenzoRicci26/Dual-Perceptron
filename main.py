import pandas as pd
import numpy as np

from DualPerceptron import DualPerceptron

def main():
    df = pd.read_csv('cmp\iris_data.csv', names = ["Sepal Lenght", "Sepal Width", "Petal Lenght", "Petal Width", "Class"])
    
    X = df.iloc[: , :-1].values
    Y = df.iloc[:, -1].values
    for i in range(Y.shape[0]):
        if Y[i] == 'Iris-setosa':
            Y[i] = -1
        else:
            Y[i] = 1
    
    """
    alpha = np.zeros(X.shape[0])

    print(alpha.shape, Y.shape)
    
    result = np.zeros(alpha.shape[0])
   
    for i in range (X.shape[0]):
        for j in range(alpha.shape[0]):
            result[j] = np.dot(X[j], X[i])
    print(result)

    print(result.shape)
    """

    model = DualPerceptron()

    model.train(X, Y, 1000)

    y_predicted = model.predict(X, Y)

    if (Y == y_predicted).all():
        print("Ce l'hai fatta Goku !!!")
       
if __name__ == "__main__":
    main()