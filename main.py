import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 

import xgboost as xgb
color_pal = sns.color_palette()
plt.style.use('fivethirtyeight')

df = pd.read_csv('PJME_hourly.csv')
df = df.set_index('Datetime')

df.plot(style='.', figsize=(15, 5), color=color_pal[0], title='PJME Enery Use in MW')
df