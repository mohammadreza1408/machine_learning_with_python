import matplotlib

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
matplotlib.use('TkAgg')


def sinplot():
    x=np.linspace(0,14,100)
    for i in range(1,7):
        plt.plot(x, np.sin(x + i * .5) * 7-i)

sinplot()
plt.show()

tip_df = pd.DataFrame(pd.read_csv("datasets/tip.csv"))
# print(tip_df)
print(tip_df.columns)

sns.displot(tip_df, x='total_bill',hue='sex',kind='kde')
sns.displot(tip_df, x='total_bill',hue='smoker',kind='kde',fill='True')
sns.color_palette()
sns.catplot(x="day", y="total_bill", data=tip_df)
plt.show()



