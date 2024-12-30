# 🤖 Dual Perceptron Binary Classifier

This project implements a Machine Learning algorithm called Dual Perceptron, designed to perform binary classification of a target attribute based on specific features. The Dual Perceptron algorithm leverages a dual representation of data, making it effective for handling linearly separable datasets.

### 🌟 Features

- Efficient binary classification using the Dual Perceptron algorithm.
- Supports customization of input features and target attributes.
- Designed with simplicity and clarity for educational and practical applications.

### 🔍 How It Works
The Dual Perceptron uses a dual representation of the input data and applies iterative updates to the weight vector to optimize the classification of the target attribute. This makes it highly effective for problems where the data is linearly separable.

# 🧠 Dual Perceptron Algorithm - Performance Report

This repository contains the implementation of the **Dual Perceptron Algorithm** for binary classification. Below are the results of tests performed on various datasets, highlighting training times, prediction times, and accuracy for different kernel types and parameters.

---

## 📊 Performance Summary  

### **1. Breast Cancer Wisconsin (Diagnostic)**  
- **Dataset**: 569 samples | 30 features  
- **Epochs**: 1000  

| **Kernel**        | **Parameter(s)**       | **Training Time (s)** | **Prediction Time (s)** | **Accuracy** |
|--------------------|------------------------|------------------------|--------------------------|--------------|
| **Linear**         | -                      | 211.10                | 0.04899                  | 0.9649       |
| **Polynomial**     | Degree=2               | 317.72                | 0.06516                  | 0.9474       |
|                    | Degree=3               | 7.74                  | 0.06485                  | 0.9561       |
|                    | Degree=4               | 13.97                 | 0.06721                  | 0.9386       |
|                    | Degree=5               | 43.87                 | 0.06633                  | 0.9211       |
| **RBF**            | Gamma=1                | 2.51                  | 0.15058                  | 0.9386       |
|                    | Gamma=2                | 2.49                  | 0.14933                  | 0.9298       |
|                    | Gamma=3                | 2.45                  | 0.15411                  | 0.9386       |
|                    | Gamma=0.1              | 4.36                  | 0.14728                  | 0.9474       |
|                    | Gamma=0.5              | 2.56                  | 0.15575                  | 0.9649       |
|                    | Gamma=0.3              | 2.47                  | 0.14887                  | 0.9649       |

---

### **2. Adult Dataset**  
- **Dataset**: 1000 samples (7 most relevant features)  
- **Epochs**: 1000  

| **Kernel**        | **Parameter(s)**       | **Training Time (s)** | **Prediction Time (s)** | **Accuracy** |
|--------------------|------------------------|------------------------|--------------------------|--------------|
| **Linear**         | -                      | 537.71                | 0.21054                  | 0.74         |
|                    | (3000 samples)         | 6540.78               | 1.47515                  | 0.7367       |
| **Polynomial**     | Degree=2               | 604.71                | 0.24014                  | 0.7367 (0.76 after fine-tuning) |
|                    | Degree=3               | 623.95                | 0.24339                  | 0.7533 (0.69 after fine-tuning) |
|                    | Degree=4               | 753.40                | 0.18601                  | 0.58         |
|                    | Degree=5               | 759.09                | 0.17594                  | 0.55         |
| **RBF**            | Gamma=1                | 2050.88               | 0.46135                  | 0.76         |
|                    | Gamma=2                | 1615.42               | 0.65630                  | 0.75         |
|                    | Gamma=3                | 1930.96               | 0.50159                  | 0.765        |
|                    | Gamma=0.1              | 3601.83               | 0.45833                  | 0.775        |

---

### **3. Rice (Cammeo or Osrick)**  
- **Dataset**: 3810 samples | 7 features  
- **Epochs**: 1000  

| **Kernel**        | **Parameter(s)**       | **Training Time (s)** | **Prediction Time (s)** | **Accuracy** |
|--------------------|------------------------|------------------------|--------------------------|--------------|
| **Linear**         | -                      | 8178.58               | 2.72629                  | 0.8705       |
| **Polynomial**     | Degree=2               | 24.11                 | 2.44595                  | 1.0          |
|                    | Degree=3               | 24.92                 | 2.61772                  | 0.9252       |
|                    | Degree=4               | 25.22                 | 2.61919                  | 1.0          |
|                    | Degree=5               | 24.43                 | 2.56026                  | 0.8031       |
| **RBF**            | Gamma=1                | 53.73                 | 5.91361                  | 1.0          |
|                    | Gamma=2                | 58.03                 | 6.94641                  | 1.0          |
|                    | Gamma=0.1              | 66.92                 | 6.66501                  | 1.0          |
|                    | Gamma=0.5              | 59.79                 | 6.90851                  | 1.0          |

---

### **4. KR-vs-KP Dataset**  
- **Dataset**: 3195 samples | 36 features  
- **Epochs**: 1000  

| **Kernel**        | **Parameter(s)**       | **Training Time (s)** | **Prediction Time (s)** | **Accuracy** |
|--------------------|------------------------|------------------------|--------------------------|--------------|
| **Linear**         | -                      | 7245.07               | 1.42522                  | 0.9687       |
| **Polynomial**     | Degree=2               | 301.67                | 2.12621                  | 0.9671       |
|                    | Degree=3               | 100.29                | 2.08110                  | 0.9703       |
|                    | Degree=4               | 77.36                 | 4.86553                  | 0.9718       |
|                    | Degree=5               | 140.83                | 1.94552                  | 0.9718       |
| **RBF**            | Gamma=1                | 91.37                 | 4.30754                  | 0.9280       |
|                    | Gamma=2                | 73.28                 | 4.36592                  | 0.9280       |
|                    | Gamma=0.1              | 117.32                | 4.29105                  | 0.9624       |
|                    | Gamma=0.2              | 169.90                | 4.35538                  | 0.9687       |
|                    | Gamma=0.15             | 86.73                 | 4.15919                  | 0.9687       |

---

## 📝 Notes  
- The **RBF Kernel** consistently shows strong accuracy across datasets with fine-tuned `gamma` values.  
- **Polynomial Kernels** perform well with lower degrees, but training time increases with complexity.  
- **Linear Kernels** are efficient for high-dimensional datasets but may underperform compared to RBF in non-linear separable datasets.  
