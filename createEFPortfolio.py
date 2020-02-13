from time import sleep
import pandas as pd

from pypfopt import expected_returns
from pypfopt import risk_models
from pypfopt.cla import CLA

from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping

import sys


def create_epochs(df, rule):
    dfSample = df.pct_change().resample(rule).mean()

    dfSample.reset_index(inplace=True)
    dfSample = dfSample[['DATE']]
    dfSample.columns = ['start']

    dfSample['end'] = dfSample.start.shift(-1)
    dfSample = dfSample[:-1]

    return dfSample


def create_xl_EFPortfolio(df, lb, ub, resample_rule):

    dfReSample = create_epochs(df, resample_rule)

    """
    # Create a dictionary of time periods (or 'epochs')
    epochs = { '0' : {'start': '1-1-2005', 'end': '31-12-2006'},
               '1' : {'start': '1-1-2007', 'end': '31-12-2008'},
               '2' : {'start': '1-1-2009', 'end': '31-12-2010'}
             }
    """
    epochs = dfReSample.to_dict('index')

    # Compute the efficient covariance for each epoch
    e_return = {}
    e_cov = {}
    efficient_portfolio = {}
    liW = []
    liR = []

    for x in epochs.keys():
        period = df.loc[epochs[x]['start']:epochs[x]['end']]

        j = (x + 1) / len(epochs.keys())
        sys.stdout.write('\r')
        sys.stdout.write("[%-20s] %d%%" % ('='*int(20*j), 100*j))
        sys.stdout.flush()

        try:
            # Compute the annualized average (mean) historical return
            # mu = expected_returns.mean_historical_return(period)#, frequency = 252)
            mu = expected_returns.ema_historical_return(period, frequency=252, span=500)

            # Compute the efficient covariance matrix
            Sigma = risk_models.CovarianceShrinkage(period).ledoit_wolf()

            # Initialize the Crtical Line Algorithm object
            efficient_portfolio[x] = CLA(mu, Sigma, weight_bounds=(lb, ub))
            efficient_portfolio[x].max_sharpe()  # min_volatility()

            cleaned_weights = efficient_portfolio[x].clean_weights()
            e_return[x] = mu
            e_cov[x] = Sigma

            liW.append(pd.DataFrame({'epochs': x, 'weights': cleaned_weights}))
            liR.append(pd.DataFrame({'epochs': x, 'returns': mu}))

        except Exception as e:
            sys.stdout.write('\r')
            sys.stdout.write('%s%s %s%s%s\n' % ('#', x, 'error:', epochs[x], e))

    dfWeightsEF = pd.concat(liW)
    dfWeightsEF.reset_index(inplace=True)
    dfWeightsEF.columns = ['asset', 'epochs', 'weights']
    dfWeightsEF = dfWeightsEF.pivot(index='epochs', columns='asset', values='weights')

    dfReturnsEF = pd.concat(liR)
    dfReturnsEF.reset_index(inplace=True)
    dfReturnsEF.columns = ['asset', 'epochs', 'returns']
    dfReturnsEF = dfReturnsEF.pivot(index='epochs', columns='asset', values='returns')

    dfWeightsEF.to_excel(r"weightsEF.xlsx")
    dfReturnsEF.to_excel(r"returnsEF.xlsx")

    dfReturns = df.pct_change().dropna()

    dfNAV = pd.merge(dfWeightsEF, dfReSample, left_index=True, right_index=True)
    dfNAV = dfNAV.drop(columns=['start'])
    dfNAV.set_index('end', inplace=True)
    dfNAV.index.names = ['DATE']
    dfNAV = dfNAV.resample('D').ffill()

    dfNAV = dfNAV.loc[dfNAV.index.isin(dfReturns.index.tolist())]
    dfReturns = dfReturns.loc[dfReturns.index.isin(dfNAV.index.tolist())]

    dfNAV['RET'] = (dfNAV.values * dfReturns.values).sum(axis=1).tolist()
    dfNAV['NAV'] = (1 + dfNAV['RET']).cumprod()

    dfNAV = pd.merge(dfNAV, dfReturns, left_index=True,
                     right_index=True, suffixes=('_weight', '_return'))

    dfNAV.to_excel(r"navEF.xlsx")

    return


def create_model(dfReturnsEF, dfWeightsEF, output_name):

    # Set the input and output data
    x = dfReturnsEF
    y = dfWeightsEF

    nVarX = len(x.columns)
    nVarY = len(y.columns)

    # Create and train the neural network with two hidden layers
    model = Sequential()
    model.add(Dense(100, input_dim=nVarX, activation='relu'))
    model.add(Dense(200, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(nVarY, activation='tanh'))  # activation='tanh'

    early_stopping_monitor = EarlyStopping(patience=2)

    model.compile(loss='mean_squared_logarithmic_error', optimizer='rmsprop', metrics=['accuracy'])
    #model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    model.fit(x, y, validation_split=0.3, epochs=100*10*1, callbacks=[early_stopping_monitor])

    model.save(output_name)

    return
