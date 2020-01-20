######################################################
# Pairs Trading Analysis with Python                 #
# AUS-RSA Pair                                       #
# (c) Diego Fernandez Garcia 2015-2018               #
# www.exfinsis.com                                   #
######################################################

# 1. Packages Importing
import numpy as np
import pandas as pd
# import pandas_datareader.data as web
import matplotlib.pyplot as plt
import statsmodels.regression.linear_model as rg
import arch.unitroot as at

##########

# 2. Pairs Trading Analysis Data

# 2.1. Data Downloading or Reading

# Data Downloading
# intquery3 = web.DataReader(['EWA', 'EZA'], 'yahoo', '2007-01-01', '2017-01-01')
# int3 = intquery1['Adj Close'].dropna()
# int3.columns = ['aus', 'rsa']

# Data Reading
data = pd.read_csv('data//Pairs-Trading-Analysis-Data.txt', index_col='Date', parse_dates=True)
int3 = data.loc[:, ['aus', 'rsa']]

# 2.2. Training and Testing Ranges Delimiting
tint3 = int3[:'2014-12-31']
tint3.columns = ['taus', 'trsa']
fint3 = int3['2015-01-02':]
fint3.columns = ['faus', 'frsa']

##########

# 3. Pairs Identification

# 3.1. AUS-RSA Returns Calculation
taus = tint3['taus']
trsa = tint3['trsa']
rtaus = taus.pct_change(1).dropna()
rtrsa = trsa.pct_change(1).dropna()

# 3.2. AUS-RSA Returns Correlation
print('')
print('== AUS-RSA Returns Correlation ==')
print('')
print(np.round(pd.DataFrame(rtaus).join(rtrsa).corr(), 4))
print('')

# 3.3. AUS-RSA Prices Chart
fig1, ax1 = plt.subplots()
ax1.plot(taus)
ax1.legend(loc='lower left')
ax2 = ax1.twinx()
ax2.plot(trsa, color='orange')
ax2.legend(loc='lower right')
plt.suptitle('AUS-RSA Prices')
plt.show()

##########

# 4. Pairs Spread Co-Integration

# 4.1. AUS-RSA Spread Calculation
# OLS regression doesn't include constant
tintsp3 = taus - rg.OLS(taus, trsa).fit().params[0] * trsa

# 4.2. AUS-RSA Spread Chart
fig2, ax = plt.subplots()
ax.plot(tintsp3, label='tintsp3')
ax.axhline(tintsp3.mean(), color='orange')
ax.legend(loc='upper left')
plt.suptitle('AUS-RSA Spread')
plt.show()

# 4.3. AUS-RSA Non-Stationary Prices
print('== AUS Prices Augmented Dickey-Fuller Test ==')
print('')
print(at.ADF(taus, trend='ct'))
print('')
print('== RSA Prices Augmented Dickey-Fuller Test ==')
print('')
print(at.ADF(trsa, trend='ct'))
print('')

# 4.4. AUS-RSA Stationary Price Differences
print('== AUS Prices Differences Augmented Dickey-Fuller Test ==')
print('')
print(at.ADF(taus.diff(1).dropna(), trend='ct'))
print('')
print('== RSA Prices Differences Augmented Dickey-Fuller Test ==')
print('')
print(at.ADF(trsa.diff(1).dropna(), trend='ct'))
print('')

# 4.5. AUS-RSA Spread Co-Integration Tests
print('== AUS-RSA Spread Augmented Dickey-Fuller Co-Integration Test ==')
print('')
print(at.ADF(tintsp3, trend='ct'))
print('')
print('== AUS-RSA Spread Phillips-Perron Co-Integration Test ==')
print('')
print(at.PhillipsPerron(tintsp3, trend='ct', test_type='rho'))
print('')
