import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

r=np.random.randint(1,100,[3,5])
print(r)

df=pd.DataFrame(data=r, index=['one','two','three'], columns=['a','b','c','d','e'])
print(df)

from new_york_time import get_new_york_time

get_time= get_new_york_time()
print(get_time)



b=np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
print(b)
print("*"*30)
print(np.reshape(b,(3,4)))
print("*"*30)
print(np.reshape(b,(1,-1)))
print("*"*30)
print(b.ndim)
print(b.dtype)
print(b.itemsize)
d1=np.arange(1,20, step=3)
print(d1)
x=np.array([[1,2],[3,4]], dtype=np.float64)
y=np.array([[5,6],[7,8]], dtype=np.float64)
print(x)
print()
print(y)

print("+"*50)
print(np.add(x,y))

print("-"*50)
print(np.subtract(x,y))

print("*"*50)
print(np.multiply(x,y))

print("/"*50)
print(np.divide(x,y))

print("radical"*50)
print(np.sqrt(x,y))

arr = np.array([2,1,5,3,7,4,6,8])
print(np.sort(arr))


a=np.array([10,20,30,40])
b=np.array([50,60,70,80])
print(np.concatenate((a,b)))
titanic_data = pd.read_csv("datasets/titanic.csv")
print(titanic_data.head())
print(titanic_data.shape)
print(titanic_data.columns)
print(titanic_data.dtypes)
print(titanic_data[['Survived']][0:5])
print(titanic_data[['Survived','Age']][10:25])
print(titanic_data[['Survived']].value_counts(normalize=True)*100)
print(pd.crosstab(titanic_data.Sex, titanic_data.Survived, normalize="index"))

print(titanic_data[titanic_data.Age <=5]["Survived"].value_counts(normalize=True))
print(titanic_data[titanic_data.Name.str.contains("Allen")])
print(titanic_data.Embarked.unique())
print(
    titanic_data[(titanic_data.Survived == 1) & (titanic_data.Age <=5)][['Age',
                                                                        'Sex',
                                                                        'Pclass']][0:5]
)



print(titanic_data[titanic_data.Age.isnull() ][['Age','Survived','Sex','Pclass']][0:5])
print(titanic_data.groupby('Pclass')['Age'].mean())

print(titanic_data.groupby(['Pclass','Sex'])['Age'].mean().reset_index())



