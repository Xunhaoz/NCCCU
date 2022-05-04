import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime

fileName = "TWSE_DLY_2330.csv"
scale = 500

prePrice = 0
history = []


def RSI(closePrice, time_period):
    global prePrice
    if len(history) == 0:
        prePrice = closePrice

    history.append(closePrice - prePrice)

    prePrice = closePrice

    if len(history) < time_period:
        return 50
    elif len(history) > time_period:
        del (history[0])

    positiveSum = 0
    allSum = 0
    for i in history:
        if i > 0:
            positiveSum += i
            allSum += i
        else:
            allSum -= i

    return (positiveSum / allSum) * 100


def toTime(timestamp):
    Date = datetime.fromtimestamp(timestamp)
    Date = Date.date()
    return Date


def preTreatment(fn):
    df = pd.read_csv(fn)[-scale:]
    df['Date'] = df['Timestamp'].apply(toTime)
    df.set_index(['Date'], inplace=True)
    df['RSI6'] = df['Close'].apply(RSI, time_period=6)
    df['RSI12'] = df['Close'].apply(RSI, time_period=12)
    df['RSI24'] = df['Close'].apply(RSI, time_period=24)
    return df


if __name__ == '__main__':
    df = preTreatment(fileName)

    img = plt.figure()
    df['RSI6'].plot(color='k', lw=1, legend=True)
    df['RSI12'].plot(color='b', lw=1, legend=True)
    df['RSI24'].plot(color='r', lw=1, legend=True)
    plt.ylim(-50, 150)
    plt.title('RSI in 2330')
    plt.xticks(rotation=45)
    plt.savefig("RSI in 2330.png")
    plt.show()
