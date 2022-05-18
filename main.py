import queue
import random

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sklearn
import yfinance as yf
from datetime import datetime
from sklearn import neighbors, datasets
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn_pandas import DataFrameMapper
from sklearn.linear_model import LogisticRegression

fileName = "TWSE_DLY_2330.csv"
scale = 2000
MACD = []
signal = []
time_period = 20
std_factor = 2
historyBB = []
smaValue = []
upperBand = []
lowerBand = []
prePrice = 0
historyRSI = []
classify = []


def MACDfun(df):
    global MACD
    global signal
    ShortEMA = df.Close.ewm(span=12, adjust=False).mean()
    LongEMA = df.Close.ewm(span=26, adjust=False).mean()
    MACD = ShortEMA - LongEMA
    signal = MACD.ewm(span=9, adjust=False).mean()
    return


def toTime(timestamp):
    Date = datetime.fromtimestamp(timestamp)
    Date = Date.date()
    return Date


def preTreatmentMACD(fn):
    df = pd.read_csv(fn)[-scale:]
    df['Date'] = df['Timestamp'].apply(toTime)
    df.set_index(['Date'], inplace=True)
    df = df.drop(['Timestamp'], axis=1)
    MACDfun(df)
    df['MACD'] = MACD
    df['signal'] = signal
    return df


def BollingerBands(df):
    s = df['Close'].count()
    for k, closePrise in enumerate(df['Close']):
        historyBB.append(closePrise)
        if len(historyBB) > time_period:
            del (historyBB[0])
        sma = np.mean(historyBB)
        smaValue.append(sma)
        std = np.sqrt(np.sum(((historyBB - sma) ** 2) / len(historyBB)))
        upperBand.append(sma + std * std_factor)
        lowerBand.append(sma - std * std_factor)
        if k >= s - 11:
            k -= 10
        if df['Close'][k + 10] > closePrise:
            classify.append(1)
        elif df['Close'][k + 10] > closePrise:
            classify.append(0)
        else:
            classify.append(-1)


def preTreatmentBB(df):
    BollingerBands(df)
    df = df.assign(Period_SMA=pd.Series(smaValue, index=df.index))
    df = df.assign(Upper_Band=pd.Series(upperBand, index=df.index))
    df = df.assign(Lower_Band=pd.Series(lowerBand, index=df.index))
    df = df.assign(Classify=pd.Series(classify, index=df.index))

    return df


def RSI(closePrice, time_period):
    global prePrice
    if len(historyRSI) == 0:
        prePrice = closePrice

    historyRSI.append(closePrice - prePrice)

    prePrice = closePrice

    if len(historyRSI) < time_period:
        return 50
    elif len(historyRSI) > time_period:
        del (historyRSI[0])

    positiveSum = 0
    allSum = 0
    for i in historyRSI:
        if i > 0:
            positiveSum += i
            allSum += i
        else:
            allSum -= i

    return (positiveSum / allSum) * 100


def preTreatmentRSI(df):
    df['RSI6'] = df['Close'].apply(RSI, time_period=6)
    df['RSI12'] = df['Close'].apply(RSI, time_period=12)
    df['RSI24'] = df['Close'].apply(RSI, time_period=24)
    return df


if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    pd.set_option('max_colwidth', 100)

    df = preTreatmentMACD(fileName)
    df = preTreatmentBB(df)
    df = preTreatmentRSI(df)

    x = df.drop(['Classify', 'Close'], axis=1)
    y = df['Classify']

    maxInvestment = 0.0
    for i in range(1, 10):
        sum = 0.0
        investment = 0.0

        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

        scaler = preprocessing.StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)

        model = neighbors.KNeighborsClassifier(n_neighbors=i, n_jobs=-1)
        model.fit(X_train, y_train)

        X_test = scaler.transform(X_test)
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        num_correct_samples = accuracy_score(y_test, y_pred, normalize=False)
        con_matrix = confusion_matrix(y_test, y_pred)

        for k, j in enumerate(y_pred):
            if j == 1:
                investment += X_test[k][0]
                if k >= 359:
                    k = 359
                    sum += X_test[k + 10][0]
                else:
                    sum += X_test[k + 10][0]
        if (sum / investment) > maxInvestment:
            maxInvestment = (sum / investment)

        print('Turnover rate:', maxInvestment)
        print('number of correct sample: {}'.format(num_correct_samples))
        print('accuracy: {}'.format(accuracy))
        print('confusion matrix:\n', con_matrix)
        print()
    print(x.shape, y.shape)
    # print(x, y)
