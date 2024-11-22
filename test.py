import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import r2_score
matplotlib.use('TkAgg')



pd.options.display.width = 0


# df = pd.read_csv("testdata.csv")
df = pd.read_csv("FuelConsumption.csv")
# print(df)

cdf=df[["ENGINESIZE","CYLINDERS","FUELCONSUMPTION_COMB","CO2EMISSIONS"]]
# print(cdf)


msk=np.random.rand(len(df))<0.8
train=cdf[msk]
test=cdf[~msk]
print(msk)
print(~msk)
print(cdf)
print(train)
print(test)

# fig= plt.figure()
# ax1=fig.add_subplot(111)
# ax1.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color='blue')
# ax1.scatter(test.ENGINESIZE, test.CO2EMISSIONS, color='red')
# plt.xlabel("ENGINESIZE")
# plt.ylabel("Emission")
# plt.show()


regr= linear_model.LinearRegression()
train_x=np.asanyarray(train[['ENGINESIZE']])
train_y=np.asanyarray(train[['CO2EMISSIONS']])
regr.fit(train_x,train_y)
print('Coefficients: ',regr.coef_)
print('intercept: ',regr.intercept_)


plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color='blue')
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')
plt.xlabel("ENGINESIZE")
plt.ylabel("Emission")
plt.show()


test_x=np.asanyarray(test[['ENGINESIZE']])
test_y=np.asanyarray(test[['CO2EMISSIONS']])
test_y_=regr.predict(test_x)

print("Mean absolote error : %.2f" % np.mean(np.absolute(test_y_ - test_y)))
print("Residual sum of squares (MSE):%2f" %np.mean((test_y_ - test_y)**2))
print("R2-score: %.2f" %r2_score(test_y, test_y_
                                 ))
