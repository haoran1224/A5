from numpy import sort
from matplotlib import pyplot
from xgboost import XGBClassifier, plot_importance, XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score, accuracy_score, mean_squared_error
import pandas as pd
import os
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
import numpy as np


os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def chuli(train_data):
    # 数据归一化处理
    from sklearn import preprocessing

    features_columns = [col for col in train_data.columns if col not in ['gen', 'time']]

    min_max_scaler = preprocessing.MinMaxScaler()
    min_max_scaler = min_max_scaler.fit(train_data[features_columns])
    train_data_scaler = min_max_scaler.transform(train_data[features_columns])
    train_data_scaler = pd.DataFrame(train_data_scaler)
    train_data_scaler.columns = features_columns
    train_data_scaler['gen'] = train_data['gen']
    return train_data_scaler


def test111():
    # load data
    dataset = pd.read_csv('data/total.csv')
    # dataset = np.array(dataset)
    # split data into X and y
    X = dataset.iloc[0:6000, 0:11]
    Y = dataset.iloc[0:6000, 11]

    # split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=7)

    # fit model on all training data
    model = XGBRegressor()
    model.fit(X_train, y_train)
    plot_importance(model)
    # make predictions for test data and evaluate
    y_pred = model.predict(X_test)
    accuracy = mean_squared_error(y_test, y_pred)
    print(accuracy)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))

    total_loss = []
    # Fit model using each importance as a threshold
    thresholds = sort(model.feature_importances_)
    for thresh in thresholds:

        # select features using threshold
        selection = SelectFromModel(model, threshold=thresh, prefit=True)
        select_X_train = selection.transform(X_train)

        # train model
        selection_model = XGBRegressor()
        selection_model.fit(select_X_train, y_train)

        # eval model
        select_X_test = selection.transform(X_test)
        y_pred = selection_model.predict(select_X_test)
        accuracy = mean_squared_error(y_test, y_pred)
        total_loss.append(accuracy)

        print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (thresh, select_X_train.shape[1], accuracy * 100.0))

    # 画图
    plt.figure(figsize=(16, 6))
    plt.plot(total_loss)
    plt.show()


if __name__ == "__main__":
    while True:
        # data = pd.read_excel('data/excel6.xlsx')
        # data = chuli(data)
        # X = data.iloc[:, 0:6]
        # Y = data.iloc[:, 6]
        # print(X.describe())
        # print(Y.head())
        # X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=30)
        #
        # model = XGBRegressor(tree_method='gpu_hist', gpu_id='0', max_depth=6, learning_rate=0.01, n_estimators=160,
        #                      silent=False, objective='reg:gamma')
        # model.fit(X_train, y_train)
        #
        # print("************************训练完成*****************")
        # # joblib.dump(model, "model.pkl")
        # # model2 = joblib.load('model.pkl')
        # y_predict = model.predict(X_test)
        # score = mean_squared_error(y_test, y_predict)
        # print("MSE:", score)
        # plot_importance(model)
        # pyplot.show()

        # clf = RandomForestRegressor(n_estimators=200)  # 200棵树模型
        # clf.fit(X_train, y_train)
        # score = mean_squared_error(y_test, clf.predict(X_test))
        # print("RandomForestRegressor:   ", score)
        # dataset = pd.read_csv('data/total.csv')
        # print(dataset.describe())
        test111()
        break


