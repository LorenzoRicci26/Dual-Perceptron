import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing, metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from DualPerceptronImpl import MyDualPerceptron
from KernelFunction import KernelFunctions


def main():

    kf = KernelFunctions()
    df = pd.read_csv('cmp/iris_data.csv', names = ["Sepal Lenght", "Sepal Width", "Petal Lenght", "Petal Width", "Class"])

    le = preprocessing.LabelEncoder()
    for i in range(df.shape[1] - 1):
        newvals = le.fit_transform(df.iloc[:, i])
        df[df.columns[i]] = newvals
    
    X = df.iloc[: , :- 1].values #Data
    y = df.iloc[:, -1].values #Target
    y = np.where ( y == 'Iris-setosa', -1, 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    #Vado a mappare il mio set di dati attraverso un metodo kernel 
    X_train_mapped = kf.rbf_kernel(X_train)
    X_test_mapped = kf.rbf_kernel(X_test)

    perceptron = MyDualPerceptron(X_train_mapped, X_test_mapped)
    
    perceptron.train(X_train, y_train)

    y_pred = perceptron.predict(X_test, y_test)

    print('Da prevedere: ' + str(y_test))
    print('Predetto: ' + str(y_pred))

    accuracy = accuracy_score(y_test, y_pred)
    print('Accuracy:', accuracy)
    
    """
    y_test = y_test.astype(int)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
    print(metrics.auc(fpr, tpr))
    print("ACCURACY:", metrics.accuracy_score(y_test, y_pred))
    """
if __name__ == "__main__":
    main()