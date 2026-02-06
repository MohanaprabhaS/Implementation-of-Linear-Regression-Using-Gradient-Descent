# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required library and read the dataframe.

2.Write a function computeCost to generate the cost function.

3.Perform iterations og gradient steps with learning rate.

4.Plot the Cost function using Gradient Descent and generate the required graph. 

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: MOHANAPRABHA S
RegisterNumber: 212224040197
*/
```
```
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def linear_regression(X1, y, learning_rate=0.1, num_iters=1000):
    X = np.c_[np.ones((len(X1), 1)), X1]   
    theta = np.zeros((X.shape[1], 1))    
    
    for _ in range(num_iters):
        predictions = X.dot(theta)
        errors = predictions - y
        theta -= learning_rate * (1 / len(X1)) * X.T.dot(errors)
    
    return theta

data = pd.read_csv("50_Startups.csv")
print(data.head())
print("\n")


X = data.iloc[:, :-1].drop(columns=['State']).values
y = data.iloc[:, -1].values.reshape(-1, 1)


scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)


theta = linear_regression(X_scaled, y_scaled)
print("Theta values:\n", theta)


new_data = np.array([[165349.2, 136897.8, 471784.1]])

new_scaled = scaler_X.transform(new_data)
new_scaled_bias = np.c_[np.ones((1, 1)), new_scaled]

prediction_scaled = new_scaled_bias.dot(theta)
prediction = scaler_y.inverse_transform(prediction_scaled)

print("\nPredicted scaled value:", prediction_scaled)
print("Predicted Profit:", prediction)
```

## Output:

<img width="798" height="179" alt="image" src="https://github.com/user-attachments/assets/babcc760-3964-417f-ab58-64e0ca66b6cb" />

<img width="616" height="203" alt="image" src="https://github.com/user-attachments/assets/0acf0f6f-c6fc-4681-9a5f-b4fd0d2f7b2e" />


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
