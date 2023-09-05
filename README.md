# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Use the standard libraries in python for Gradient Design.
2. Set variables for assigning dataset values.
3. Import linear regression from sklearn.
4. Assign the points for representing the graph.
5. Predict the regression for marks by using the representation of the graph.
6. Compare the graphs and hence we obtained the linear regression for the given data.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: PRADEEPASRI S
RegisterNumber: 212221220038
*/
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('/content/student_scores.csv')
#displaying the content in datafile
print("df.head():")
df.head()

print("df.tail():")
df.tail()

#Segregating data to variables
print("Array value of X:")
X=df.iloc[:,:-1].values
X

print("Array value of X:")
Y=df.iloc[:,1].values
Y

#splitting train and test data
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)

#displaying predicted values
print("Values of Y prediction:")
Y_pred

#displaying actual values
print("Array values of Y test:")
Y_test

#graph plot for training data
print("Training set graph:")
plt.scatter(X_train,Y_train,color="orange")
plt.plot(X_train,regressor.predict(X_train),color="red")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

#graph plot for test data
print("Test set graph:")
plt.scatter(X_test,Y_test,color="purple")
plt.plot(X_test,regressor.predict(X_test),color="yellow")
plt.title("Hours vs Scores (Test Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

print("Values of MSE,MAE and RMSE:")
mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)

mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)

rmse=np.sqrt(mse)
print('RMSE = ',rmse)
print('Values of MSE')
```

## Output:
![simple linear regression model for predicting the marks scored](sam.png)

![Screenshot (86)](https://github.com/pradeepasri26/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/131433142/d5fa1fcc-0d58-4046-a262-b0a05d648b22)

![Screenshot (87)](https://github.com/pradeepasri26/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/131433142/601b2eed-a170-4b5e-ae41-68de05bd6ba8)

![Screenshot (88)](https://github.com/pradeepasri26/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/131433142/f9896738-4752-49db-8c1b-0d79841de7d8)

![Screenshot (89)](https://github.com/pradeepasri26/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/131433142/243c0ed0-2610-43bf-9833-15a4d1813e8b)

![Screenshot (90)](https://github.com/pradeepasri26/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/131433142/f498acb8-792f-4c6e-891c-a4d56240548a)

![Screenshot (91)](https://github.com/pradeepasri26/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/131433142/7a2ce631-7e73-4edc-9072-2d7f46a8347f)

![Screenshot (92)](https://github.com/pradeepasri26/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/131433142/095e40f3-e637-44c5-af9d-9052c28eff36)

![Screenshot (93)](https://github.com/pradeepasri26/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/131433142/bb56f1d2-be34-4ba8-a30b-9b5d0d65881b)

![Screenshot (94)](https://github.com/pradeepasri26/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/131433142/1bc53e38-05ce-4235-954f-dd42e0d9582a)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
