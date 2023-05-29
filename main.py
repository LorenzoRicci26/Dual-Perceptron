import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest, chi2
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
        scaler = StandardScaler()
        df[numeric_features] = scaler.fit_transform(df[numeric_features])

        X = df.iloc[:, [0] + list(range(2, len(df.columns)))].values #Data
        y = df.iloc[:, 1].values #Target
        y = np.where(y == 'M', -1, 1)

    elif df_type == '2':

        df = pd.read_csv('cmp\_adult.csv')

        #Inizializzo 
        le = LabelEncoder()

        #Trasformo le colonne in dati numerici 
        for col in df.columns[:-1]:
            if df[col].dtype == 'object':
                df[col] = le.fit_transform(df[col])

        # Gestisci i valori mancanti
        numeric_features = df.select_dtypes(include=['int64', 'float64']).columns
        df[numeric_features] = df[numeric_features].fillna(df[numeric_features].mean())

        X_selected = df.iloc[:, :-1].values
        y = df.iloc[: , -1].values
        y = np.where( y == '<=50K', -1, 1)

        # Applica la selezione delle caratteristiche
        selector = SelectKBest(score_func=chi2, k=10)  # Scegli il numero di caratteristiche da selezionare
        X = selector.fit_transform(X_selected, y)

        # Normalizza le caratteristiche selezionate
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        print(X,X.shape)

    elif df_type == '3':

        df = pd.read_csv('cmp\heart_disease.csv', sep= ',')

        #Normalizzo le caratteristiche numeriche 
        numeric_features = df.select_dtypes(include=['int64', 'float64']).columns
        scaler = StandardScaler()
        df[numeric_features] = scaler.fit_transform(df[numeric_features])

        X = df.iloc[: , :-1].values
        y = df.iloc[: , -1].values
        y = np.where( y == 0, -1, 1)
        print(X,X.shape)

    elif df_type == '4':
        
        df = pd.read_csv('cmp\_rice_and_cammeo.csv')

        #Normalizzo le caratteristiche numeriche 
        numeric_features = df.select_dtypes(include=['int64', 'float64']).columns
        scaler = StandardScaler()
        df[numeric_features] = scaler.fit_transform(df[numeric_features])

        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        y = np.where(y == 'Cammeo', -1, 1)
        print(X,X.shape)

    else:
        print('Hai inserito un numero sbagliato !!!')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42) #Imposto il test_size a 0 perchè tanto non mi serve la matrice di test

    #Input per scegliere quale tipo di mappatura kernel di vuole utilizzare
    print('Quale tipo di mappatura kernel desideri utilizzare? | 1: linear,  2: polynomial,  3: rbf')
    kernel_type = input('Scegli 1, 2 o 3 : ')

    #Inserisco all'interno della matrice kernelizzata per il train i valori a seconda del tipo di kernel scelto
    if kernel_type == '1':
        kernel = KernelFunctions().linear_kernel
    elif kernel_type == '2':
        kernel = KernelFunctions().polynomial_kernel
    elif kernel_type == '3':
        kernel = KernelFunctions().rbf_kernel
    else:
        print('Hai inserito un input non corretto !!!')

    perceptron = MyDualPerceptron()

    start2 = timeit.default_timer()
    a, b = perceptron.train(X_train, y_train, kernel, epochs=1000)
    end2 = timeit.default_timer()
    print(f"Time taken to train the data is {end2 - start2}s")

    # Valutazione delle performance sul set di test
    start3 = timeit.default_timer()
    y_test_pred = perceptron.predict(X_test, X_train, y_train, a, b, kernel)
    end3 = timeit.default_timer()
    print(f"Time taken to predict the target is {end3 - start3}s")
    accuracy_test = accuracy_score(y_test, y_test_pred)
    print("Accuracy sul set di test:", accuracy_test)

if __name__ == "__main__":
    main()