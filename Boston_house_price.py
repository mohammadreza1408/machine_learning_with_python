import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
matplotlib.use('TkAgg')


boston_df= pd.DataFrame(pd.read_csv('E:/DATA_SETS/BostonHousing.csv'))
print(boston_df.columns)

boston_df['PRICE']= boston_df.medv
print(boston_df.head())
# print(boston_df.shape)
# print(boston_df.dtypes)
# print(boston_df.isnull().sum())
# print(boston_df.describe())
# corr = boston_df.corr()
# plt.matshow(boston_df.corr())
# plt.show()


