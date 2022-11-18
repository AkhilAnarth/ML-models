import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix , classification_report
df = pd.read_csv('Breast cancer data.csv')
print(df.shape)
print(df.head(5).to_string())

# Working with certain columns
x = df.iloc[: , 2:6]
print(x.head().to_string())
y = df.iloc[: , 1]
print(y.head().to_string())

#training data
x_train = x[:300]
y_train = y[:300]

#test data
x_test = x[300:500]
y_test = y[300:500]

print(y_test.to_list())

#create model
Logmodel = LogisticRegression()
Logmodel.fit(x_train , y_train)
y_predict = Logmodel.predict(x_test)
print(y_predict.tolist())

#confusion matrix
con_matrix = confusion_matrix(y_test , y_predict)
print(con_matrix)

#print report
print(classification_report(y_test , y_predict))

