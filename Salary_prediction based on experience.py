import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error , mean_absolute_error
import matplotlib.pyplot as plt

df = pd.read_csv('Salary_Data.csv')
print(df.head())

# Data preprocessing
x = df.iloc[: , :-1]
y = df.iloc[: , 1]

# Splitting the dataset
x_train , x_test , y_train , y_test = train_test_split(x , y , test_size=1/3 , random_state=0)

# creating the model
Linmodel = LinearRegression()

# Training the model
Linmodel.fit(x_train,y_train)

# predicting the results
y_pred = Linmodel.predict(x_test)
print(y_pred)

# Visualising the results

# Plot for train
plt.scatter(x_train,y_train,color='red') #plotting the observation line
plt.plot(x_train,Linmodel.predict(x_train),color='blue') #plotting regression line
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('salary')
plt.show()

# Plot for test
plt.scatter(x_test,y_test,color='red')
plt.plot(x_test,Linmodel.predict(x_test),color='blue')
plt.title('Salary vs Experience (Testing Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Intercept
intercept = Linmodel.intercept_
print('Intercept:',intercept)

# Slope
slope = Linmodel.coef_
print('Slope:',slope)

experience = 10
salary = slope * experience + intercept
print('Salary:',salary)

# MAE (Mean absolute error)
mae = mean_absolute_error(y_test,y_pred)
print(mae)

# Mean squared error
mse = mean_squared_error(y_test,y_pred)
print(mse)

# Rooted mean squared error
rmse = np.sqrt(mse)
print(rmse)



