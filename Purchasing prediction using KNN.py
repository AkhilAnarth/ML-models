import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix , classification_report
from sklearn.model_selection import train_test_split

df = pd.read_csv('Social_Network_Ads.csv')
print(df.head())

x = df.iloc[: , 2:4]
y = df.iloc[: , -1]
print(x , y)

x_train , x_test , y_train , y_test = train_test_split(x,y, test_size=0.3 , random_state=0)

knn = KNeighborsClassifier(n_neighbors=4 , metric='minkowski' , p = 2 )
knn.fit(x_train,y_train)
y_pred = knn.predict(x_test)

con_mat = confusion_matrix(y_test,y_pred)
print(con_mat)
print(classification_report(y_test,y_pred))



