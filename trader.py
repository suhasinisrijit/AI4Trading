import numpy as np
import pandas as pd
from os import path

import pandas_datareader as dr
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

from scipy.spatial import distance
import matplotlib.pyplot as plt
from tabulate import tabulate

from statsmodels.tsa.api import adfuller
from statsmodels.tsa.stattools import coint

sp500MemberListFile = 'data/sp500_constituents.csv'
sp500MemberListFundamentalFile = 'data/sp500_constituents_financials.csv'
sp500PickleLocation = 'data/sp500data.pkl'
symbolPriceDataLocation = 'data/symbol/'


def getDailyPrice(tickers, start, end):
    print('Fetching data for {}'.format(tickers))
    return dr.get_data_yahoo(tickers, start, end)['Adj Close']


def readSavedPriceData(ticker):
    return pd.read_pickle('{}{}.pkl'.format(symbolPriceDataLocation, ticker))


def readSavedPriceDataForTickers(tickerList):
    retDF = pd.DataFrame()
    for i in range(len(tickerList)):
        tickerPrice = readSavedPriceData(tickerList[i]).reset_index()
        if (i == 0):
            retDF = tickerPrice
        else:
            retDF = retDF.merge(tickerPrice, on='Date', how='outer')
    retDF = retDF.set_index('Date')
    return retDF

def getPreprocessedSP500Data():
    sp500Fundamentals = pd.read_csv(sp500MemberListFundamentalFile)
    return preprocessSP500Fundamentals(sp500Fundamentals)

def getSP500Details(start, end):
    sp500DF = pd.DataFrame()
    #sp500ComponentDf = pd.read_csv(sp500MemberListFile)
    sp500ComponentDf = pd.read_csv(sp500MemberListFundamentalFile)
    sp500SymbolList = sp500ComponentDf['symbol'].tolist()

    successCount = 0
    for symbol in sp500SymbolList:
        try:
            tickerQuote = dr.get_quote_yahoo(symbol)
            getDailyPrice([symbol], start, end).to_pickle('{}{}.pkl'.format(symbolPriceDataLocation, symbol))
            sp500DF = sp500DF.append(tickerQuote, sort=False)
            successCount += 1
        except Exception as e:
            print('Error {} for symbol {}'.format(e, symbol))

    print('Successfully retrieved data for {} stocks'.format(successCount))
    print(sp500DF)
    sp500DF.to_pickle(sp500PickleLocation)


def readSP500Data(requiredDataFields):
    sp500DFFromPickle = pd.read_pickle(sp500PickleLocation)
    """
    Columnns are : 'language', 'region', 'quoteType', 'triggerable', 'quoteSourceName',
       'currency', 'priceHint', 'sharesOutstanding', 'bookValue',
       'fiftyDayAverage', 'fiftyDayAverageChange',
       'fiftyDayAverageChangePercent', 'twoHundredDayAverage',
       'twoHundredDayAverageChange', 'twoHundredDayAverageChangePercent',
       'marketCap', 'forwardPE', 'priceToBook', 'sourceInterval',
       'exchangeDataDelayedBy', 'tradeable', 'exchange', 'shortName',
       'longName', 'messageBoardId', 'exchangeTimezoneName',
       'exchangeTimezoneShortName', 'gmtOffSetMilliseconds', 'market',
       'esgPopulated', 'firstTradeDateMilliseconds', 'postMarketChangePercent',
       'postMarketTime', 'postMarketPrice', 'postMarketChange',
       'regularMarketChange', 'regularMarketChangePercent',
       'regularMarketTime', 'regularMarketPrice', 'regularMarketDayHigh',
       'regularMarketDayRange', 'regularMarketDayLow', 'regularMarketVolume',
       'regularMarketPreviousClose', 'bid', 'ask', 'bidSize', 'askSize',
       'fullExchangeName', 'financialCurrency', 'regularMarketOpen',
       'averageDailyVolume3Month', 'averageDailyVolume10Day',
       'fiftyTwoWeekLowChange', 'fiftyTwoWeekLowChangePercent',
       'fiftyTwoWeekRange', 'fiftyTwoWeekHighChange',
       'fiftyTwoWeekHighChangePercent', 'fiftyTwoWeekLow', 'fiftyTwoWeekHigh',
       'dividendDate', 'earningsTimestamp', 'earningsTimestampStart',
       'earningsTimestampEnd', 'trailingAnnualDividendRate', 'trailingPE',
       'trailingAnnualDividendYield', 'marketState', 'epsTrailingTwelveMonths',
       'epsForward', 'price'    
    """

    sp500DF = sp500DFFromPickle[requiredDataFields]
    sp500DF.index.name = 'symbol'
    sp500DF.reset_index(inplace=True)
    sp500DF = sp500DF.fillna(0)
    return sp500DF

def preprocessSP500Fundamentals(sp500Fundamentals):
    toKeep = [x for x in sp500Fundamentals['symbol'] if path.exists('{}{}.pkl'.format(symbolPriceDataLocation, x))]
    sp500Fundamentals = sp500Fundamentals[sp500Fundamentals['symbol'].isin(toKeep)]

    sectorEncoding = pd.get_dummies(sp500Fundamentals['sector'])
    df = sp500Fundamentals.join(sectorEncoding)
    #df = df.drop(['name','sector','price','52weeklow','52weekhigh','pricesalesratio','pricebookratio'],axis=1)
    df = df.drop(['name', 'sector', '52weeklow', '52weekhigh'], axis=1)
    colsToNormalize = ['price', 'peratio','dividendyield','marketcap','eps','ebitda','pricesalesratio','pricebookratio']
    for colToNormalize in colsToNormalize:
        df[colToNormalize] = df.loc[:, df.columns==colToNormalize].apply(lambda x: x / x.max())

    df.columns = [x.lower().replace(' ','_') for x in df.columns]
    #print(tabulate(df,tablefmt='psql',headers=df.columns))
    return df

def performKMeanClustering(data, numClusters):
    data = data.fillna(0)
    kMeans = KMeans(n_clusters=numClusters)
    kMeans.fit(data)
    return kMeans


def findK(data, start, stop):
    errors = []
    k = range(start, stop)
    print('Analyzing {} to {} clusters '.format(start, stop))
    for i in k:
        kMeans = performKMeanClustering(data, i)
        dist = distance.cdist(data, kMeans.cluster_centers_, 'euclidean')
        dist = [d for d in dist if not np.isnan(np.min(d))]
        mindist = np.nanmin(dist, axis=1)
        errors.append(sum(mindist) / data.shape[0])

    plt.plot(k, errors, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k')
    plt.show()


def getClustersForK(data, numClusters):
    kMeans = performKMeanClustering(data.drop('symbol', axis=1), numClusters)
    data['cluster'] = kMeans.labels_
    groupedData = data.groupby('cluster')
    return groupedData


def checkCointegration(s1, s2, ticker1, ticker2):
    s1.columns = ['price']
    s2.columns = ['price']
    #returnS1 = s1.pct_change(1).dropna()
    #returnS2 = s2.pct_change(1).dropna()
    returnS1 = s1.dropna()
    returnS2 = s2.dropna()

    """
    reg = LinearRegression().fit(returnS1, returnS2)
    beta = float(reg.coef_.squeeze())
    spread = returnS2 - (beta * returnS1)
    
    """
    lookBack = 2
    tempDF = pd.DataFrame()
    tempDF['X']=s1['price']
    tempDF['Y']=s2['price']

    tempDF['cov']=tempDF.rolling(lookBack).cov().unstack()['X']['Y']
    tempDF['var']=tempDF['Y'].rolling(lookBack).var()
    tempDF['beta']=tempDF['cov']/tempDF['var']
    tempDF['spread']=tempDF['Y'] - (tempDF['X'] * tempDF['beta'])

    spread = tempDF['spread'].dropna()
    # cointPValue = adfuller(spread['price'], regression='c')
    score, cointPValue, _ = coint(returnS1, returnS2)
    #print("Cointegration test result for {} and {} is {}".format(ticker1, ticker2, cointPValue))
    return (np.round(cointPValue, 5), np.round(1, 4), spread)

def getPriceAndFindCointegration(ticker1, ticker2):
    if (ticker1 == ticker2):
        return (np.Inf, 0)
    else:
        try:
            priceTicker1 = readSavedPriceData(ticker1)
            priceTicker2 = readSavedPriceData(ticker2)

            if (priceTicker1.shape[0] != priceTicker2.shape[0]):
                commonDates = priceTicker1.index.intersection(priceTicker2.index)
                priceTicker1 = priceTicker1.loc[commonDates]
                priceTicker2 = priceTicker2.loc[commonDates]

            return checkCointegration(priceTicker1, priceTicker2, ticker1, ticker2)
        except Exception as e:
            print("Error {}".format(e))
            return (np.Inf, 0)


def calculateCointegration(dfCol):
    ticker1 = dfCol.name
    return [getPriceAndFindCointegration(ticker1, ticker2) for ticker2 in dfCol.index]


def plotStockPrices(tickerList):
    tickerPrices = readSavedPriceDataForTickers(tickerList)
    tickerPrices.plot(kind='line')
    plt.show()


def findPairs(listStocks, treshold):
    tickers = listStocks.tolist()
    tickers.append('SPY')
    cointegratedPairs = []
    for i in range(len(tickers)):
        for j in range(i + 1, len(tickers)):
            s1 = tickers[i]
            s2 = tickers[j]
            checkCointResult = getPriceAndFindCointegration(s1, s2)
            if (checkCointResult[0] <= treshold):
                cointegratedPairs.append((s1, s2, checkCointResult))
                # print('Cointegration for {} and {} is {}'.format(s1,s2,checkCointResult[0]))

    tickerRelatedToSPY = [ p[1] if p[0]=='SPY' else p[0] for p in cointegratedPairs if p[0]=='SPY' or p[1]=='SPY']
    print(tickerRelatedToSPY)

    return cointegratedPairs

def pairTradeSP500(numClusters,start, end):
    treshold = 0.01
    groupedData = getClustersForK(getPreprocessedSP500Data(), numClusters)

    for i, cluster in groupedData:
        print(cluster.shape)
        if (cluster.shape[0] >= 2 and cluster.shape[0] <= 100):
            print('Finding pairs in cluster {} with {} stocks'.format(i, cluster.shape[0]))
            cointegratedPairs = findPairs(cluster['symbol'], treshold)
            if len(cointegratedPairs) > 0:
                pairedTickers = [[s1, s2] for s1, s2, p in cointegratedPairs]
                tickersToPlot = list(set([ticker for tickers in pairedTickers for ticker in tickers]))
                #plotStockPrices(tickersToPlot)
                print('Pairs for cluster {} is {}'.format(i, [(s1, s2) for s1, s2, p in cointegratedPairs]))
            else:
                print('No pairs found for cluster {}'.format(i))


start = "01/01/2019"
end = "12/01/2019"
#s1= readSavedPriceData('MSFT')
#s2=readSavedPriceData('AAPL')
#checkCointegration(s1,s2,'MSFT','AAPL')

#getSP500Details(start,end)
#sp500DataPreprocessed = getPreprocessedSP500Data()
#findK(sp500DataPreprocessed.drop('symbol', axis=1),2,50)
k = 30
pairTradeSP500(k,start,end)