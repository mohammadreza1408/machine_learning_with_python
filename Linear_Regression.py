import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

x= np.array([1,2,16,4,5]) #independent variable
y= np.array([2,4,5,4,5]) #dependent variable

N=len(x)

m= (N * np.sum(x*y) - np.sum(x) * np.sum(y)) / (N* np.sum(x **2) - np.sum(x **2))
c= (np.sum(y) - m * np.sum(x))/ N

print(f"Slop (m) is : {m}")
print(f"intercept elevation (c) is : {c}")


y_pred = m * x +c
plt.scatter(x,y,color='blue',label='Data')
plt.plot(x,y_pred, color='red',label='Regretion Line')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()