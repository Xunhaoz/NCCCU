import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

fileName = "TWSE_DLY_2330.csv"
scale = 500
MACD = []
signal = []

def MACDfun():
    global MACD
    global signal
    ShortEMA = df.Close.ewm(span=12, adjust=False).mean()
    LongEMA = df.Close.ewm(span=26, adjust=False).mean()
    MACD = ShortEMA - LongEMA
    signal = MACD.ewm(span=9, adjust=False).mean()
    return 1

def toTime(timestamp):
    Date = datetime.fromtimestamp(timestamp)
    Date = Date.date()
    return Date


def preTreatment(fn):
    df = pd.read_csv(fn)[-scale:]
    df['Date'] = df['Timestamp'].apply(toTime)
    df.set_index(['Date'], inplace=True)
    return df


if __name__ == '__main__':
    df = preTreatment(fileName)
    MACDfun()
    plt.plot(df.index, MACD, label='MACD', color='r')
    plt.plot(df.index, signal, label='signal', color='b')
    plt.legend(loc='best')
    plt.title('MACD and SIGNAL in 2330')
    plt.savefig("MACD and SIGNAL in 2330.png")
    plt.xticks(rotation=45)
    plt.show()
