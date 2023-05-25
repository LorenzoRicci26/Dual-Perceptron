import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import timeit

from DualPerceptronImpl import MyDualPerceptron
from KernelFunction import KernelFunctions


def main():
    
    #Input per scegliere quale dataset si desidera utilizzare
    print('Quale dataSet desideri utilizzare? | 1: Breast Cancer Wisconsin (Diagnostic),  2: Adult,  3: Heart Disease,  4: Rice (Cammeo and Osmancik)')
    df_type = input('Scegli 1, 2, 3 o 4 : ')

    if df_type == '1':

        df = pd.read_csv('cmp\wdbc_data.csv')
        
        #Normalizzo le caratteristiche numeriche 
        numeric_features = df.select_dtypes(include=['int64', 'float64']).columns
        scaler = preprocessing.StandardScaler()
        df[numeric_features] = scaler.fit_transform(df[numeric_features])

        X = df.iloc[: , 2:].values #Data
        y = df.iloc[:, 1].values #Target
        y = np.where(y == 'M', -1, 1)

    elif df_type == '2':

        df = pd.read_csv('cmp\_adult_data.csv')

        #Inizializzo 
        le = preprocessing.LabelEncoder()

        #Trasformo le colonne in dati numerici 
        for i in range(df.shape[1] - 1):
            newvals = le.fit_transform(df.iloc[:, i])
            df[df.columns[i]] = newvals

        #Normalizzo le caratteristiche numeriche 
        numeric_features = df.select_dtypes(include=['int64', 'float64']).columns
        scaler = preprocessing.StandardScaler()
        df[numeric_features] = scaler.fit_transform(df[numeric_features])

        X = df.iloc[: , :-1].values
        y = df.iloc[: , -1].values
        y = np.where( y == '<=50K', -1, 1)

        print(X)

    elif df_type == '3':

        df = pd.read_csv('cmp\heart.csv')

        #Normalizzo le caratteristiche numeriche 
        numeric_features = df.select_dtypes(include=['int64', 'float64']).columns
        scaler = preprocessing.StandardScaler()
        df[numeric_features] = scaler.fit_transform(df[numeric_features])

        X = df.iloc[: , :-1].values
        y = df.iloc[: , -1].values
        y = np.where( y == 0, -1, 1)
        print(X)

    elif df_type == '4':
        
        df = pd.read_csv('cmp\_rice.csv')

        #Normalizzo le caratteristiche numeriche 
        numeric_features = df.select_dtypes(include=['int64', 'float64']).columns
        scaler = preprocessing.StandardScaler()
        df[numeric_features] = scaler.fit_transform(df[numeric_features])

        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        y = np.where(y == 'Cammeo', -1, 1)
        print(X,X.shape)

    else:
        print('Hai inserito un numero sbagliato !!!')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30) #Imposto il test_size a 0 perchè tanto non mi serve la matrice di test

    #Input per scegliere quale tipo di mappatura kernel di vuole utilizzare
    print('Quale tipo di mappatura kernel desideri utilizzare? | 1: linear,  2: polynomial,  3: rbf')
    kernel_type = input('Scegli 1, 2 o 3 : ')
        
    perceptron = MyDualPerceptron(kernel_type)
    
    start2 = timeit.default_timer()
    perceptron.train(X_train, y_train, epochs=1500)
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

if __name__ == "__main__":
    main()