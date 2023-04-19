import math

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import torch.nn as nn
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

input_window = 10
output_window = 1
batch_size = 10  # batch size
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def chuli(train_data):
    # 数据归一化处理
    from sklearn import preprocessing
    features_columns = [col for col in train_data.columns if col not in ['gen', 'Time']]
    min_max_scaler = preprocessing.MinMaxScaler()
    min_max_scaler = min_max_scaler.fit(train_data[features_columns])
    train_data_scaler = min_max_scaler.transform(train_data[features_columns])
    train_data_scaler = pd.DataFrame(train_data_scaler)
    train_data_scaler.columns = features_columns
    train_data_scaler['gen'] = train_data['gen']
    return train_data_scaler


def te(train_data_scaler):
    # 特征相关性
    plt.figure(figsize=(20, 16))
    column = train_data_scaler.columns.tolist()
    mcorr = train_data_scaler[column].corr(method="spearman")
    mask = np.zeros_like(mcorr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    g = sns.heatmap(mcorr, mask=mask, cmap=cmap, square=True, annot=True, fmt='0.2f')
    plt.show()


def getdata1():
    from pandas import read_csv
    gen_data = pd.read_excel('../data/excel6.xlsx', usecols=['gen'])
    other_data = pd.read_excel('../data/excel6.xlsx', usecols=['temperature', 'apparentTemperature'])
    # 归一化处理
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    gen_data = scaler.fit_transform(gen_data.to_numpy().reshape(-1, 1)).reshape(-1)
    other_data = chuli(other_data)


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
    # time = np.arange(0, 400, 0.1)
    # amplitude = np.sin(time) + np.sin(time * 0.05) + np.sin(time * 0.12) * np.random.normal(-0.2, 0.2, len(time))
    #
    # from pandas import read_csv
    # series = read_csv('daily-min-temperatures.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
    # 读取gen这一列数据
    from pandas import read_csv
    data = read_csv('../data/Australia.csv', usecols=['电力负荷'])
    # 归一化
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    # amplitude = scaler.fit_transform(data.to_numpy().reshape(-1, 1)).reshape(-1)
    amplitude = data.to_numpy()
    # amplitude = scaler.fit_transform(amplitude.reshape(-1, 1)).reshape(-1)
    # 划分训练测试集
    sampels = 3000
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
    feature = read_csv('../data/Australia.csv',
                       usecols=['month', 'day', '小时', '干球温度', '露点温度', '湿球温度', '湿度', '电价'])
    feature = chuli(feature)
    feature = np.array(feature)

    feature_train = feature[:sampels]
    feature_test = feature[sampels:4000]

    feature_train = creat_inout_sq(feature_train, input_window)
    feature_train = feature_train[:-output_window]

    feature_test = creat_inout_sq(feature_test, input_window)
    feature_test = feature_test[:-output_window]

    return train_sequence, test_data, feature_train, feature_test


def get_batch(source, i, batch_size):
    seq_len = min(batch_size, len(source) - 1 - i)
    data = source[i:i + seq_len]
    # print(torch.stack([item[0] for item in data]).shape)
    # print(torch.stack([item[0] for item in data]).chunk(input_window, 1))
    # 1 按列切分
    input = torch.stack(torch.stack([item[0] for item in data]).chunk(input_window, 1))  # 1 is feature size
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

# 显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

if __name__ == "__main__":
    while True:
        # train_data, val_data, train_data_fe, val_data_fe = get_data()
        # # 测试数据部分
        # print(train_data_fe.shape)
        # need_data = train_data[2970:2980]
        # # 2895,2,100,tensor
        # print(train_data.shape)
        # print(need_data.shape[0])
        #
        # sorce, target = get_batch(need_data, 0, 1)
        # print(torch.as_tensor(sorce).shape)
        data = pd.read_excel('temp.xlsx')
        plt.plot(np.array(data.loc[0]), label="原数据")
        plt.plot(np.array(data.loc[1]), label="LSTM-多变量")
        plt.plot(np.array(data.loc[2]), label="LSTM-单变量")
        plt.plot(np.array(data.loc[3]), label="Transformer-单变量")
        plt.plot(np.array(data.loc[4]), label="Transformer-多变量")

        plt.grid()
        plt.legend()
        plt.show()
        break
