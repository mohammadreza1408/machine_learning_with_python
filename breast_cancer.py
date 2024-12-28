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
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


#load dataset
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()

x_train,x_test,y_train,y_test =train_test_split(cancer.data,cancer.target,random_state=42)

print("Shape of Train Data: ",x_train.shape)
print("Shape of Test Data: ",x_test.shape)

# sns.countplot(cancer.target)
# plt.show()
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier()
knn.fit(x_train,y_train)
y_pred=knn.predict(x_test)
print("Acuracy of Tran data : ",knn.score(x_train,y_train))
print("Acuracy of Test data : ",knn.score(x_test,y_test))

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# print(classification_report(y_test,y_pred))
# cm = confusion_matrix(y_test,y_pred)
# sns.heatmap(cm,square=True,annot=True)
# plt.show()

err_rate=[]
for i in range(1,40):
    knn=KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train,y_train)
    pred_i=knn.predict(x_test)
    err_rate.append(np.mean(pred_i !=y_test))

# plt.figure(figsize=(10,6))
# plt.plot(range(1,40),err_rate,color='blue',linestyle='dashed',marker='o',markerfacecolor='red', markersize=10)
# plt.title('Error Rate vs. K Value')
# plt.xlabel('K')
# plt.ylabel('Error Rate')
# plt.show()


knn=KNeighborsClassifier(n_neighbors=11)
knn.fit(x_train,y_train)
y_pred=knn.predict(x_test)
print(knn.score(x_train,y_train))
print(knn.score(x_test,y_test))
