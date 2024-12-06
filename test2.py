import matplotlib

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
matplotlib.use('TkAgg')
# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)

tip_df = pd.DataFrame(pd.read_csv("datasets/tip.csv"))

print(tip_df.columns)
print(tip_df.info())
print(tip_df.describe())
print(tip_df.sample(5))
print(tip_df.groupby('day').count())

df2=tip_df.groupby('day').sum()
df2.drop('size', inplace=True, axis =1)
df2['persent']= df2['tip']/df2['total_bill']*100
print(df2)



df3=tip_df.groupby('smoker').sum()
df3['persent'] = df3['tip']/df3['total_bill'] *100
print(df3)

df4=tip_df.groupby(['day','size']).sum()
df4['percent']=df4['tip']/df4['total_bill'] *100
df4.dropna()
print(df4)

tip_df.replace({'sex':{'Male':0 , 'Female':1}, 'smoker':{'No':0, 'Yes': 1}}, inplace=True)
# print(tip_df.head())

days= pd.get_dummies(tip_df['day'])
# print(days.sample(5))
df=pd.concat([tip_df,days],axis=1)
times = pd.get_dummies(tip_df['time'])
df=pd.concat([tip_df,times], axis=1)
print(df.columns)
# print(df)
x=df[['sex','smoker','size','Fri','Sat','Sun','Dinner']]
y=df[['tip']]


# print(tip_df.groupby('day').count())
df2=tip_df.groupby('day').sum()
df2.drop('size', inplace = True, axis=1)
df2['persent'] = df2['tip']/df2['total_bill'] * 100
print(df2)

print(sns.catplot(x='day', kind='count',data=tip_df))
plt.show()
