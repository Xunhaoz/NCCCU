import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime

fileName = "TWSE_DLY_2330.csv"
scale = 500
time_period = 20
std_factor = 2
# SMA simple moving Average
history = []
smaValue = []
upperBand = []
lowerBand = []


def toTime(timestamp):
    Date = datetime.fromtimestamp(timestamp)
    Date = Date.date()
    return Date


def preTreatment(fn):
    df = pd.read_csv(fn)[-scale:]
    df['Date'] = df['Timestamp'].apply(toTime)
    df.set_index(['Date'], inplace=True)

    for closePrise in df['Close']:
        history.append(closePrise)
        if len(history) > time_period:
            del (history[0])
        sma = np.mean(history)
        smaValue.append(sma)
        std = np.sqrt(np.sum(((history - sma) ** 2) / len(history)))
        upperBand.append(sma + std * std_factor)
        lowerBand.append(sma - std * std_factor)

    df = df.assign(Close_prise=pd.Series(df['Close'], index=df.index))
    df = df.assign(Period_SMA=pd.Series(smaValue, index=df.index))
    df = df.assign(Upper_Band=pd.Series(upperBand, index=df.index))
    df = df.assign(Lower_Band=pd.Series(lowerBand, index=df.index))

    return df


if __name__ == '__main__':
    df = preTreatment(fileName)

    img = plt.figure()
    df['Close_prise'].plot(color='k', lw=1, legend=True)
    df['Period_SMA'].plot(color='b', lw=1, legend=True)
    df['Upper_Band'].plot(color='r', lw=1, legend=True)
    df['Lower_Band'].plot(color='g', lw=1, legend=True)
    plt.show()
