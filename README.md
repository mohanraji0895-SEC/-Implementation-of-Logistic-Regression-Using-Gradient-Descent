# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the dataset and drop unwanted columns.
2. Convert categorical columns to numeric codes.
3. Separate features (X) and target (y), initialize theta, then run gradient descent using the sigmoid function to update weights.
4. Predict labels using the trained theta, compute accuracy, and predict for new students.

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: MOHANRAJI D
RegisterNumber: 212225060164
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('Placement_Data.csv')

print("Original Data:")
print(data.head())

data = data.drop('sl_no', axis=1)
data = data.drop('salary', axis=1)

print("\nAfter dropping 'sl_no' and 'salary':")
print(data.head())

data["gender"] = data["gender"].astype('category')
data["ssc_b"] = data["ssc_b"].astype('category')
data["hsc_b"] = data["hsc_b"].astype('category')
data["degree_t"] = data["degree_t"].astype('category')
data["workex"] = data["workex"].astype('category')
data["specialisation"] = data["specialisation"].astype('category')
data["status"] = data["status"].astype('category')
data["hsc_s"] = data["hsc_s"].astype('category')

print("\nData types after converting to 'category':")
print(data.dtypes)

data["gender"] = data["gender"].cat.codes
data["ssc_b"] = data["ssc_b"].cat.codes
data["hsc_b"] = data["hsc_b"].cat.codes
data["degree_t"] = data["degree_t"].cat.codes
data["workex"] = data["workex"].cat.codes
data["specialisation"] = data["specialisation"].cat.codes
data["status"] = data["status"].cat.codes
data["hsc_s"] = data["hsc_s"].cat.codes

print("\nData after converting categories to numeric codes:")
print(data.head())

x = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

print("\nFeature matrix X shape:", x.shape)
print("Target vector y shape:", y.shape)

theta = np.random.randn(x.shape[1])

print("\nInitial theta (weights):")
print(theta)

def sigmoid(z):
return 1 / (1 + np.exp(-z))

def loss(theta, X, y):
h = sigmoid(X.dot(theta))
return -np.sum(y * np.log(h + 1e-10) + (1 - y) * np.log(1 - h + 1e-10))

def gradient_descent(theta, X, y, alpha, num_iterations):
m = len(y)
for i in range(num_iterations):
h = sigmoid(X.dot(theta))
gradient = X.T.dot(h - y) / m
theta -= alpha * gradient
if (i + 1) % 100 == 0:
current_loss = loss(theta, X, y)
print(f"Iteration {i+1}, Loss: {current_loss:.4f}")
return theta

theta = gradient_descent(theta, x, y, alpha=0.01, num_iterations=1000)

print("\nFinal theta (weights) after training:")
print(theta)

def predict(theta, X):
h = sigmoid(X.dot(theta))
y_pred = np.where(h >= 0.5, 1, 0)
return y_pred

y_pred = predict(theta, x)
accuracy = np.mean(y_pred.flatten() == y)

print("\nTraining Accuracy:", accuracy)
print("Predicted labels (first 20):")
print(y_pred[:20])

xnew1 = np.array([[0, 87, 0, 95, 0, 2, 78, 2, 0, 0, 1, 0]])
xnew2 = np.array([[0, 0, 0, 0, 0, 2, 8, 2, 0, 0, 1, 0]])

y_prednew1 = predict(theta, xnew1)
y_prednew2 = predict(theta, xnew2)

print("\nPrediction for new student 1 (0 = Not Placed, 1 = Placed):", y_prednew1[0])
print("Prediction for new student 2 (0 = Not Placed, 1 = Placed):", y_prednew2[0])


## Output:
## Output:
![logistic regression using gradient descent](sam.png)


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.


