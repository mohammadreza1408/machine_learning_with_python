import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

x= np.array([1,2,16,4,5]) #independent variable
y= np.array([2,4,5,4,5]) #dependent variable




m,c =0,0
learning_rate = 0.01
epochs = 1000

for _ in range(epochs):
    y_pred = m * x +c

    dm = (-2 / len(x)) * np.sum(x * (y- y_pred))
    dc = (-2 / len(x)) * np.sum(y - y_pred)


    m -= learning_rate *dm
    c -= learning_rate * dc

print(f"Slop (m): {m}")
print(f"intercept elevation (c): {c}")

plt.scatter(x, y, color="blue", label="Data")
plt.plot(x, m * x + c, color="green", label="Gradient descent")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()