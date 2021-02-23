import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import numpy as np

df_A = pd.read_csv('./Data/BTC-USD.csv')
df_B = pd.read_csv('./Data/DOT1-USD.csv')
df_C = pd.read_csv('./Data/ETH-USD.csv')
df_D = pd.read_csv('./Data/TETHER-USD.csv')

df_A['Date'] = pd.to_datetime(df_A['Date'])
df_A.tail()

plt.figure(figsize = (15,10))
plt.plot(df_A['Date'], df_A['Close'], label='Bitcoin')
#plt.plot(df_B['Date'], df_B['Close'], label='Polkadot')
#plt.plot(df_C['Date'], df_C['Close'], label='Etherium')
#plt.plot(df_D['Date'], df_D['Close'], label='Tether')
plt.legend(loc='best')
plt.show()
