import numpy as np
import pandas as pd
import pandas_datareader  as dr
from statsmodels.tsa.api import adfuller

def test1():
    s = np.asarray([1, 2, 3, 4, 2])

    d = {}
    e={}
    for x in s:
        d[x] = [x for x, y in enumerate(s, 1)]
    print(d)

    for k, v in d.items():
        print(k,v)
        e[k] = np.asarray(v)*k + 5

    print(e)
    df = pd.DataFrame.from_dict(e, orient='index', columns=s)
    print(df)

def test2():
    def f1(a, b):
        return [(a,j*a+5) for j in b]

    s = [1, 2, 3, 4, 3]

    df = pd.DataFrame(columns=s, index=s)
    df = df.apply(lambda x: f1(x.name, x.index))
    print(df)
    dd=df.apply(lambda x: min([y[1] for y in x]))

    print(dd)


def testDF():

    data1 = {'id':['a','b','f'], 'qty1':[1,2,3]}

    data2 = {'id':['a','b','e'], 'qty2':[4,5,6]}

    pd1=pd.DataFrame(data1).set_index('id')
    pd2=pd.DataFrame(data2).set_index('id')

    ix1 = pd1['qty1'].index
    ix2=pd2['qty2'].index

    ix3= ix1.intersection(ix2)

    print(pd1)
    print(pd2)
    print('--------------')
    print(pd1.loc[ix3])
    print('--------------')
    print(pd2.loc[ix3])
    print('--------------')

    pd1=pd1.reset_index()
    pd2=pd2.reset_index()

    newDF=pd1.merge(pd2,on='id',how="outer")
    newDF.set_index('id',inplace=True)
    print(newDF)

l1=['a','b','c']
l2=['c','d','e']
print(set(l1)-set(l2))

#testDF()