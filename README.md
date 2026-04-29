# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries and load the dataset.
2. Initialize weights and bias, then apply the sigmoid function.
3. Train the logistic regression model using gradient descent by updating weights and bias.
4. Predict the output class and evaluate the model accuracy.


## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: VASHMITHA V
RegisterNumber: 212225240180 
*/
# Logistic Regression using Gradient Descent

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Step 1: Create sample dataset
data = pd.DataFrame({
    'x1': [2, 4, 6, 8, 10, 12, 14, 16],
    'x2': [1, 3, 5, 7, 9, 11, 13, 15],
    'y':  [0, 0, 0, 0, 1, 1, 1, 1]
})

print("Dataset Preview:")
print(data)

# Step 2: Select features and target
X = data[['x1', 'x2']].values
y = data['y'].values

# Step 3: Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Step 4: Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Step 5: Initialize parameters
weights = np.zeros(X_train.shape[1])
bias = 0
learning_rate = 0.01
epochs = 1000

# Step 6: Gradient Descent Training
for i in range(epochs):
    linear_model = np.dot(X_train, weights) + bias
    y_pred = sigmoid(linear_model)

    dw = (1 / len(X_train)) * np.dot(X_train.T, (y_pred - y_train))
    db = (1 / len(X_train)) * np.sum(y_pred - y_train)

    weights -= learning_rate * dw
    bias -= learning_rate * db

# Step 7: Prediction
linear_test = np.dot(X_test, weights) + bias
y_pred_test = sigmoid(linear_test)
y_pred_class = [1 if i > 0.5 else 0 for i in y_pred_test]

# Step 8: Accuracy
print("\nPredicted Values:", y_pred_class)
print("Actual Values:   ", y_test.tolist())
print("Accuracy:", accuracy_score(y_test, y_pred_class))
```

## Output:
<img width="1885" height="923" alt="image" src="https://github.com/user-attachments/assets/7b6b34c2-1ad5-4146-b79d-f881e192ea5d" />




## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

