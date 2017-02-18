

import numpy as np
import pandas as pd
import seaborn as sns

df = pd.read_csv('./training.csv')

def movingaverage(interval, window_size):
    window = np.ones(int(window_size)) / float(window_size)
    return np.convolve(interval, window, 'same')


a_moves = movingaverage(df['moves'], 50)
print(a_moves.shape)
print(len(df))
df['a_moves'] = a_moves

import matplotlib.pyplot as plt

sns.set_style('darkgrid')
df.plot(x='epochs', y=['moves', 'a_moves'])
plt.show()
