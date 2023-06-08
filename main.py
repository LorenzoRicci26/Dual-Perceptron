import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest, chi2
import timeit
# Libraries for data visualization
import matplotlib.pyplot as pplt 
from pandas.plotting import scatter_matrix

from DualPerceptronImpl import MyDualPerceptron
from KernelFunction import KernelFunctions

def main():
    
    #Input per scegliere quale dataset si desidera utilizzare
    print('Quale dataSet desideri utilizzare? | 1: Breast Cancer Wisconsin (Diagnostic),  2: Adult,  3: Rice (Cammeo and Osmancik),  4: King-Rook vs King-pawn')
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
        
        df.rename(columns={'capital-gain': 'capital gain', 'capital-loss': 'capital loss', 'native-country': 'country','hours-per-week': 'hours per week','marital-status': 'marital'}, inplace=True)
        
        #Finding the special characters in the data frame 
        df.isin(['?']).sum(axis=0)
        
        # code will replace the special character to nan and then drop the columns 
        df['country'] = df['country'].replace('?',np.nan)
        df['workclass'] = df['workclass'].replace('?',np.nan)
        df['occupation'] = df['occupation'].replace('?',np.nan)
        df.dropna(how='any',inplace=True)
            
        #dropping based on uniqueness of data from the dataset 
        df.drop(['educational-num','age', 'hours per week', 
                'fnlwgt', 'capital gain','capital loss', 'country'], axis=1, inplace=True)

        #mapping the data into numerical data using map function
        df['income'] = df['income'].map({'<=50K': -1, '>50K': 1}).astype(int)

        #gender
        df['gender'] = df['gender'].map({'Male': 0, 'Female': 1}).astype(int)
        #race
        df['race'] = df['race'].map({'Black': 0, 'Asian-Pac-Islander': 1, 'Other': 2, 'White': 3, 'Amer-Indian-Eskimo': 4}).astype(int)
        #marital
        df['marital'] = df['marital'].map({'Married-spouse-absent': 0, 'Widowed': 1, 'Married-civ-spouse': 2, 'Separated': 3, 'Divorced': 4,
                                        'Never-married': 5, 'Married-AF-spouse': 6}).astype(int)
        #workclass
        df['workclass'] = df['workclass'].map({'Self-emp-inc': 0, 'State-gov': 1,'Federal-gov': 2, 'Without-pay': 3, 'Local-gov': 4,'Private': 5, 'Self-emp-not-inc': 6}).astype(int)
        #education
        df['education'] = df['education'].map({'Some-college': 0, 'Preschool': 1, '5th-6th': 2, 'HS-grad': 3, 'Masters': 4, '12th': 5, 
                                    '7th-8th': 6, 'Prof-school': 7,'1st-4th': 8, 'Assoc-acdm': 9, 'Doctorate': 10, '11th': 11,'Bachelors': 12, '10th': 13,'Assoc-voc': 14,'9th': 15}).astype(int)
        #occupation
        df['occupation'] = df['occupation'].map({ 'Farming-fishing': 1, 'Tech-support': 2, 'Adm-clerical': 3, 'Handlers-cleaners': 4, 
                            'Prof-specialty': 5,'Machine-op-inspct': 6, 'Exec-managerial': 7,'Priv-house-serv': 8,'Craft-repair': 9,'Sales': 10, 'Transport-moving': 11, 
                            'Armed-Forces': 12, 'Other-service': 13,'Protective-serv':14}).astype(int)
        #relationship
        df['relationship'] = df['relationship'].map({'Not-in-family': 0, 'Wife': 1, 'Other-relative': 2, 'Unmarried': 3,'Husband': 4,'Own-child': 5}).astype(int)

        X = df.iloc[:1000, :-1].values
        y = df.iloc[:1000, -1].values

        print(X)

    elif df_type == '3':

        df = pd.read_csv('cmp\_rice_and_cammeo.csv')

        #Normalizzo le caratteristiche numeriche 
        numeric_features = df.select_dtypes(include=['int64', 'float64']).columns
        scaler = StandardScaler()
        df[numeric_features] = scaler.fit_transform(df[numeric_features])

        X = df.iloc[: , :-1].values
        y = df.iloc[: , -1].values
        y = np.where( y == 0, -1, 1)
        
        print(X,X.shape)

    elif df_type == '4':
        
        df = pd.read_csv('cmp\kr-vs-kp.csv')

        # Rimuovi eventuali righe o colonne vuote o mancanti
        df.dropna(inplace=True)

        # Trasforma l'ultima colonna in -1 per 'nowin' e 1 per 'won'
        df[df.columns[-1]] = df[df.columns[-1]].apply(lambda x: -1 if x == 'nowin' else 1)

        label_encoder = LabelEncoder()
        for col in df.select_dtypes(include='object'):
            df[col] = label_encoder.fit_transform(df[col])

        scaler = StandardScaler()
        df[df.columns[:-1]] = scaler.fit_transform(df[df.columns[:-1]]) 

        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values

        print(X,X.shape)

    else:
        print('Hai inserito un numero sbagliato !!!')
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42) 

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