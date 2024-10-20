import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 

import xgboost as xgb
color_pal = sns.color_palette()
plt.style.use('fivethirtyeight')

df = pd.read_csv('PJME_hourly.csv')
df = df.set_index('Datetime')

#df.plot(style='.', figsize=(15, 5), color=color_pal[0], title='PJME Enery Use in MW')
pd.to_datetime(df.index)

#split the data into 2 parts from 2008 to 2015, for training the model; and from 2015 to 2018 for testin the model
train = df.loc[df.index < '2015-01-01']
test = df.loc[df.index >= '2015-01-01']


#one single week of energy consumption
df.loc[(df.index > '2010-01-01') & (df.index < '2010-01-08')].plot(figsize=(15, 5), title='Week of Data')
plt.show()