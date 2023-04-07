import math

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import torch.nn as nn
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

input_window = 72
output_window = 24
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


def getdata():
    from pandas import read_csv
    gen_data = pd.read_excel('../data/excel6.xlsx',usecols=['gen'])
    other_data = pd.read_excel('../data/excel6.xlsx',usecols=['temperature','apparentTemperature'])
    # 归一化处理
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    gen_data = scaler.fit_transform(gen_data.to_numpy().reshape(-1, 1)).reshape(-1)
    other_data = chuli(other_data)




