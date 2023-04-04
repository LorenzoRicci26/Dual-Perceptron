import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing, metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import timeit
import time 

from DualPerceptronImpl import MyDualPerceptron
from KernelFunction import KernelFunctions


def main():

    start = timeit.default_timer()

    #Inizializzo 
    kf = KernelFunctions()
    le = preprocessing.LabelEncoder()
    
    #Input per scegliere quale dataset si desidera utilizzare
    print('Quale dataSet desideri utilizzare? | 1: Whine-Quality-Red,  2: Adult,  3: Bank')
    df_type = input('Scegli 1,2 o 3 : ')

    if df_type == '1':

        df = pd.read_csv('cmp\winequality-red.csv', sep=';')

        X = df.iloc[: , :- 1].values #Data
        y = df.iloc[:, -1].values #Target
        y = np.where ( y < 5, -1, 1)

    elif df_type == '2':

        df = pd.read_csv('cmp\_adult_data.csv')

        #Trasformo le colonne in dati numerici 
        for i in range(df.shape[1] - 1):
            newvals = le.fit_transform(df.iloc[:, i])
            df[df.columns[i]] = newvals

        X = df.iloc[:5000 , :-1].values
        y = df.iloc[:5000 , -1].values
        y = np.where( y == '<=50K', -1, 1)

    elif df_type == '3':

        df = pd.read_csv('cmp\_bank-additional.csv', sep=';')

        #Trasformo le colonne in dati numerici 
        for i in range(df.shape[1] - 1):
            newvals = le.fit_transform(df.iloc[:, i])
            df[df.columns[i]] = newvals

        X = df.iloc[: , :-1].values
        y = df.iloc[: , -1].values
        y = np.where( y == 'no', -1, 1)

    else:
        print('Hai inserito un numero sbagliato !!!')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    print('Data_train: ' + str(X_train) + str(X_train.shape))
    print('Data_test: ' + str(X_test) + str(X_test.shape))
    print('Target_train: ' + str(y_train) + str(y_train.shape))
    print('Target_test: ' + str(y_test) + str(y_test.shape))

    #Input per scegliere quale tipo di mappatura kernel di vuole utilizzare
    print('Quale tipo di mappatura kernel desideri utilizzare? | 1: linear,  2: polynomial,  3: rbf')
    kernel_type = input('Scegli 1, 2 o 3 : ')
    
    #Vado a mappare il mio set di dati attraverso il metodo kernel selezionato dall'utente
    start1 = timeit.default_timer()
    if kernel_type == '1':
        X_train_mapped = kf.linear_kernel(X_train)
        X_test_mapped = kf.linear_kernel(X_test)
    elif kernel_type == '2':
        X_train_mapped = kf.polynomial_kernel(X_train, degree=3)
        X_test_mapped = kf.polynomial_kernel(X_test, degree=3)
    elif kernel_type == '3':
        X_train_mapped = kf.rbf_kernel(X_train)
        X_test_mapped = kf.rbf_kernel(X_test)
    else:
        print('Hai inserito un numero sbagliato !!!')
    
    end1 = timeit.default_timer()
    print(f"Time taken to map the data with the Kernel method choosen is {end1 - start1}s")
        
    perceptron = MyDualPerceptron(X_train_mapped, X_test_mapped)
    
    start2 = timeit.default_timer()
    perceptron.train(X_train, y_train)
    end2 = timeit.default_timer()
    print(f"Time taken to train the data is {end2 - start2}s")
    
    start3 = timeit.default_timer()
    y_pred = perceptron.predict(X_test, y_test)
    end3 = timeit.default_timer()
    print(f"Time taken to predict the target is {end3 - start3}s")

    print('Da prevedere: ' + str(y_test))
    print('Predetto: ' + str(y_pred))

    accuracy = accuracy_score(y_test, y_pred)
    print('Accuracy:', accuracy)

    end = timeit.default_timer()
    print(f"Time taken to run the the entire program is {end - start}s")

if __name__ == "__main__":
    main()