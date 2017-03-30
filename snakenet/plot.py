

import numpy as np
import pandas as pd
import seaborn as sns

df = pd.read_csv('./log_training.csv')

a_moves = df['moves'].rolling(window=50).mean()
df['a_moves'] = a_moves

a_food = df['food'].rolling(window=50).mean()
df['a_food'] = a_food

a_av = df['mean_val'].rolling(window=50).mean()
df['a_mean_val'] = a_av

import matplotlib.pyplot as plt


sns.set_style('darkgrid')
f, axarr = plt.subplots(nrows=3, ncols=1, sharex=True, sharey=False)

axarr[0].plot(df['epochs'], df['moves'], c='blue')
axarr[0].plot(df['epochs'], df['a_moves'], c='red')
axarr[0].set_title('Moves Per Game')

axarr[1].plot(df['epochs'], df['food'], c='blue')
axarr[1].plot(df['epochs'], df['a_food'], c='red')
axarr[1].set_title('Food Per Game')

axarr[2].semilogy(df['epochs'], df['mean_val'], c='blue')
axarr[2].semilogy(df['epochs'], df['a_mean_val'], c='red')
axarr[2].set_title('Mean Action Value Per Game')


plt.show()
