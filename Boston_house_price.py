import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
matplotlib.use('TkAgg')
from sklearn.linear_model import LinearRegression
from sklearn import metrics




boston_df= pd.DataFrame(pd.read_csv('E:/DATA_SETS/BostonHousing.csv'))
print(boston_df.columns)

boston_df['PRICE']= boston_df.medv
# print(boston_df.head())


x= boston_df.drop(['PRICE'], axis=1)
y=boston_df['PRICE']
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=4)
mlr = LinearRegression()
mlr.fit(X_train,y_train)

print("mlr.intercept : ",mlr.intercept_)
print("mlr.coef : ", mlr.coef_)

coeffcients = pd.DataFrame([X_train.columns, mlr.coef_]).T
coeffcients = coeffcients.rename(columns={0: 'Attribute', 1: 'Coefficents'})
# print(coeffcients)

y_pred=mlr.predict(X_train)
print('R^2: ',metrics.r2_score(y_train,y_pred))
print('MAE: ', metrics.mean_absolute_error(y_train,y_pred))
print('MSE: ',metrics.mean_squared_error(y_train,y_pred))
print('RMSE :',np.sqrt(metrics.mean_squared_error(y_train,y_pred)))

plt.scatter(y_train,y_pred)
plt.xlabel("Price")
plt.ylabel("Predicted price")
plt.title("Price vs Predicted price")
plt.show()