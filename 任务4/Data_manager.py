import MySQLdb
import inline as inline
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from scipy import stats
import warnings
import os

from sklearn.preprocessing import MinMaxScaler

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

warnings.filterwarnings("ignore")

input_window = 100
output_window = 5
batch_size = 10  # batch size
scaler = MinMaxScaler(feature_range=(-1, 1))


def chuli(train_data):
    # 数据归一化处理
    from sklearn import preprocessing
    features_columns = [col for col in train_data.columns if col not in ['gen', 'Time']]
    min_max_scaler = preprocessing.MinMaxScaler()
    min_max_scaler = min_max_scaler.fit(train_data[features_columns])
    train_data_scaler = min_max_scaler.transform(train_data[features_columns])
    train_data_scaler = pd.DataFrame(train_data_scaler)
    train_data_scaler.columns = features_columns
    # train_data_scaler['gen'] = train_data['gen']
    return train_data_scaler


# if window is 100 and prediction step is 1
# in -> [0..99]
# target -> [1..100]
def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L - tw):
        train_seq = np.append(input_data[i:i + tw][:-output_window], output_window * [0])
        train_label = input_data[i:i + tw]
        # train_label = input_data[i+output_window:i+tw+output_window]
        inout_seq.append((train_seq, train_label))
    return torch.FloatTensor(inout_seq)


# 取出相应的特征
def creat_inout_sq(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L - tw):
        train_label = input_data[i:i + tw]
        inout_seq.append(train_label)
    return torch.FloatTensor(inout_seq)


def get_data():
    from pandas import read_csv
    data = read_csv('D:\pyhtonProject\A5\任务4\data\Australia.csv', usecols=['电力负荷'])
    # 归一化
    global scaler
    amplitude = scaler.fit_transform(data.to_numpy().reshape(-1, 1)).reshape(-1)
    # amplitude = scaler.fit_transform(amplitude.reshape(-1, 1)).reshape(-1)
    # 划分训练测试集
    sampels = 2800
    train_data = amplitude[:sampels]
    test_data = amplitude[sampels:4000]

    # convert our train data into a pytorch train tensor
    # train_tensor = torch.FloatTensor(train_data).view(-1)
    # todo: add comment..
    train_sequence = create_inout_sequences(train_data, input_window)
    train_sequence = train_sequence[:-output_window]  # todo: fix hack?

    # test_data = torch.FloatTensor(test_data).view(-1)
    test_data = create_inout_sequences(test_data, input_window)
    test_data = test_data[:-output_window]  # todo: fix hack?

    # 读取特征数据 用于拼接
    feature = read_csv('D:\pyhtonProject\A5\任务4\data\Australia.csv',
                       usecols=['month', 'day', '小时', '干球温度', '露点温度', '湿球温度', '湿度', '电价'])
    feature = chuli(feature)
    feature = np.array(feature)

    feature_train = feature[:sampels]
    feature_test = feature[sampels:4000]

    feature_train = creat_inout_sq(feature_train, input_window)
    feature_train = feature_train[:-output_window]

    feature_test = creat_inout_sq(feature_test, input_window)
    feature_test = feature_test[:-output_window]

    return train_sequence.to(device), test_data.to(device), feature_train.to(device), feature_test.to(device)


def get_batch(source, i, batch_size):
    seq_len = min(batch_size, len(source) - 1 - i)
    data = source[i:i + seq_len]

    # 1 按列切分
    input = torch.stack(torch.stack([item[0] for item in data]).chunk(input_window, 1))
    target = torch.stack(torch.stack([item[1] for item in data]).chunk(input_window, 1))
    return input, target


# 此时返回的特征是可以直接与位置编码后512维的电力数据相加的
# 此时feature的维度为 100,10,2
def get_batch_tezhen(feature, i, batch_size):
    seq_len = min(batch_size, len(feature) - 1 - i)
    data = feature[i:i + seq_len]
    feature = torch.stack(torch.chunk(data, input_window, 1))
    # 分割过后特征维度会加1，进行降维
    feature = torch.squeeze(feature, 2)
    return feature


def bianma():
    data = pd.read_excel('data/Australia.xlsx')
    # data = data.drop(['time'], axis=1)
    print(data.head())
    # dumpies = pd.get_dummies(data,columns=['icon','summary'])
    # print(dumpies[:5])
    # dumpies.to_excel('data/excel2.xlsx',index=False)
    data.to_csv('data/Australia.csv', index=False)


# 查看数据分布情况
def look_data():
    data = pd.read_csv('data/temp3.csv', usecols=['gen'])
    train_data = data[:4000]
    test_data = data[3000:4000]

    plt.figure(figsize=(16, 6))
    # sns.boxplot(train_data[:],orient='v',width=0.5)
    plt.plot(train_data)
    plt.show()

    # data = pd.read_excel('data/澳大利亚.xlsx', usecols=['电力负荷'])
    # print(data.info())
    # plt.plot(data.iloc[0:17520])
    # plt.show()


if __name__ == "__main__":
    while True:
        # mysql = MySQLdb.connect(host='localhost', port=3306, user='root', passwd='123456', db='electricity')
        # sql = 'SELECT * from homea,homea_meter where homea.time = homea_meter.Time'
        # data = pd.read_sql(sql,con=mysql)
        # data.to_excel('data/total.xlsx', sheet_name='Sheet1', index=False)
        bianma()
        # look_data()
        break
