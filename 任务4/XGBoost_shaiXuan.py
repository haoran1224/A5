
from matplotlib import pyplot
from xgboost import XGBClassifier, plot_importance, XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score, accuracy_score, mean_squared_error
import pandas as pd
import os

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


if __name__ == "__main__":
    while True:
        data = pd.read_excel('data/excel6.xlsx')
        data = chuli(data)
        X = data.iloc[:, 0:6]
        Y = data.iloc[:, 6]
        print(X.describe())
        print(Y.head())
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=30)

        model = XGBRegressor(tree_method='gpu_hist', gpu_id='0', max_depth=6, learning_rate=0.01, n_estimators=160,
                             silent=False, objective='reg:gamma')
        model.fit(X_train, y_train)

        print("************************训练完成*****************")
        # joblib.dump(model, "model.pkl")
        # model2 = joblib.load('model.pkl')
        y_predict = model.predict(X_test)
        score = mean_squared_error(y_test, y_predict)
        print("MSE:", score)
        plot_importance(model)
        pyplot.show()

        # clf = RandomForestRegressor(n_estimators=200)  # 200棵树模型
        # clf.fit(X_train, y_train)
        # score = mean_squared_error(y_test, clf.predict(X_test))
        # print("RandomForestRegressor:   ", score)
        break
