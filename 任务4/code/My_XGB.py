import pandas as pd
from numpy import asarray
from pandas import DataFrame
from sklearn.metrics import mean_absolute_error,mean_squared_error,mean_absolute_percentage_error
from matplotlib import pyplot
from pandas import concat
from xgboost import XGBRegressor

import lightgbm as lgb


# transform a time series dataset into a supervised learning dataset
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols = list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
    # put it all together
    # 横向拼接
    agg = concat(cols, axis=1)
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg.values


# split a univariate dataset into train/test sets
def train_test_split(data, n_test):
    return data[:-n_test, :], data[-n_test:, :]


# fit an xgboost model and make a one step prediction
def xgboost_forecast(train, testX):
    # transform list into array
    train = asarray(train)
    # split into input and output columns
    trainX, trainy = train[:, :-1], train[:, -1]
    # fit model
    model = XGBRegressor(objective='reg:squarederror', n_estimators=1000)
    model.fit(trainX, trainy)
    # make a one-step prediction
    yhat = model.predict(asarray([testX]))
    return yhat[0]


# walk-forward validation for univariate data
def walk_forward_validation(data, n_test):
    predictions = list()
    # split dataset
    train, test = train_test_split(data, n_test)
    # seed history with training dataset
    history = [x for x in train]
    # step over each time-step in the test set
    for i in range(len(test)):
        # split test row into input and output columns
        testX, testy = test[i, :-1], test[i, -1]
        # fit model on history and make a prediction
        yhat = xgboost_forecast(history, testX)
        # store forecast in list of predictions
        predictions.append(yhat)
        # add actual observation to history for the next loop
        history.append(test[i])
        # summarize progress
        print('>expected=%.1f, predicted=%.1f' % (testy, yhat))
    # estimate prediction error
    error = mean_absolute_error(test[:, -1], predictions)
    error2 = mean_squared_error(test[:, -1], predictions)
    error3 = mean_absolute_percentage_error(test[:, -1], predictions)
    return error, error2, error3,test[:, -1], predictions


if __name__ == "__main__":
    while True:
        data = pd.read_csv('../data/Australia.csv', usecols=['电力负荷'])
        data = data[0:3000]
        print(data.describe())
        data = series_to_supervised(data, n_in=6)
        print(data.shape)
        # evaluate
        mae, MSE, MAPE,y, yhat = walk_forward_validation(data, 12)
        print('MAE: %.3f' % mae)
        print('MSE: %.6f' % MSE)
        print('MAPE: %.6f' % MAPE)
        # plot expected vs preducted
        pyplot.plot(y, label='Expected')
        pyplot.plot(yhat, label='Predicted')
        pyplot.legend()
        pyplot.show()

        break
