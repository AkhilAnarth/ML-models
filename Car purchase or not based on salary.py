import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix , classification_report
from sklearn.model_selection import train_test_split

df = pd.read_csv('car_purchase.csv')
print(df.head().to_string())
print(df.isnull().sum())

new_df = df.drop (columns=['User ID','Gender'],axis=1)
print(new_df.head().to_string())

#df['Gender'] = df['Gender'].map({'Male':1,'Female':0})
#print(df)

# Splitting the df into 2 parameters
x = new_df.iloc[:,0:2]
y = new_df.iloc[:, 2]
print(x)
print(y)

# Splitting the dataset
x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=1/3,random_state=0)

# creating the model
logmodel = LogisticRegression()

# training the model
logmodel.fit(x_train,y_train)

# predicting the outcome
y_predict = logmodel.predict(x_test)
print(y_predict.tolist())

# confusion matrix
con_mat = confusion_matrix(y_test,y_predict)
print(con_mat)

#classification report
print(classification_report(y_test,y_predict))
