import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn

from sklearn.model_selection import train_test_split


matplotlib.use('TkAgg')
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.preprocessing import PolynomialFeatures



iris_df= pd.DataFrame(pd.read_csv('E:/DATA_SETS/iris.csv'))
print(iris_df.columns)
# print(iris_df.head())

print(iris_df['variety'].value_counts())
print('*'*50)
# print(iris_df.info())

# g= sns.pairplot(iris_df, hue='variety', markers='+')
# plt.show()


X=iris_df.drop(['variety'], axis=1)
y=iris_df['variety']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.4, random_state=5)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

logreg =LogisticRegression()
logreg.fit(X_train,y_train)

y_pred=logreg.predict(X_test)

print('The accuracy of Logestic Regression is :',metrics.accuracy_score(y_test,y_pred))