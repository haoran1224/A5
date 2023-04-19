import joblib

# 模型
import torch
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler
from torch import nn
import numpy as np
import os
import pandas as pd

from 任务4.Data_manager import chuli
from 任务4.code.MyTransformer import TransAm

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from 任务4.code.MyLSTM import LSTM

input_window = 100
output_window = 1
scaler = MinMaxScaler(feature_range=(-1, 1))

truth_data = None
predict_data = None

# 预测存在问题
def predict_future(eval_model, data, steps):
    eval_model.eval()
    total_loss = 0.
    test_result = torch.Tensor(0)
    truth = torch.Tensor(0)
    # _, data = get_batch(data_source, 0, 1)
    with torch.no_grad():
        for i in range(0, steps, 1):
            input = torch.clone(data[-input_window:])
            input[-output_window:] = 0
            output = eval_model(data[-input_window:], data[-1:].T)
            data = torch.cat((data, output[-1:]))

    data = data.cpu().view(-1)

    pyplot.plot(data[-input_window-steps:], color="red")
    pyplot.plot(data[-input_window-steps:-steps], color="blue")
    pyplot.grid(True, which='both')
    pyplot.axhline(y=0, color='k')
    pyplot.show()
    # pyplot.savefig('../image/transformer-future%d.png' % steps)
    pyplot.close()

    return data


# 预测存在问题
def predict_future_duo(eval_model, data, fe, steps):
    eval_model.eval()
    total_loss = 0.
    test_result = torch.Tensor(0)
    truth = torch.Tensor(0)
    # _, data = get_batch(data_source, 0, 1)
    with torch.no_grad():
        for i in range(0, steps, 1):
            input = torch.clone(data[-input_window:])
            input[-output_window:] = 0
            output = eval_model(data[-input_window:], fe[3000 - input_window + i:3000 + i])
            data = torch.cat((data, output[-1:]))

    data = data.cpu().view(-1)

    pyplot.plot(data[-input_window-steps:], color="red")
    pyplot.plot(data[-input_window-steps:-steps], color="blue")
    pyplot.grid(True, which='both')
    pyplot.axhline(y=0, color='k')
    pyplot.show()
    # pyplot.savefig('../image/transformer-future%d.png' % steps)
    pyplot.close()

    return data


def get_kehu_data():
    data = pd.read_csv('data/Australia.csv', usecols=['电力负荷'])
    data = np.array(data)
    global truth_data
    truth_data = data[3000:3100]
    data = data[0:3000, :]

    print(data.shape)
    # 归一化
    amplitude = scaler.fit_transform(np.array(data).reshape(-1, 1)).reshape(-1)
    # 转换为张量
    data = torch.FloatTensor(amplitude)
    # 转换为（100，batch_size,维度）
    data = data.view(len(data), 1, -1).to(device)

    return data


def get_kehu_fe():
    feature = pd.read_csv('data/Australia.csv', usecols=['month', 'day', '小时', '干球温度', '露点温度', '湿球温度', '湿度', '电价'])
    feature = feature.loc[0:3999, :]
    feature = chuli(feature)
    feature = np.array(feature)
    # 转换为张量
    data = torch.FloatTensor(feature)
    # 转换为（100，batch_size,维度）
    data = data.reshape(data.shape[0], 1, 8).to(device)
    return data


# LSTM-单变量
def get_model_LSTM():
    model = LSTM(1, 64, 2, 1)
    model.load_state_dict(torch.load('save/LSTM_model.pt'))
    model.to(device)
    return model


# TRANSFORMER-单变量
def get_model_Transformer():
    model = TransAm()
    model.load_state_dict(torch.load('save/TRANSFORMER_model.pt'))
    model.to(device)
    return model


# TRANSFORMER-多变量
def get_model_Transformer_Duo():
    model = TransAm()
    model.load_state_dict(torch.load('save/TRANSFORMER_model_Duo.pt'))
    model.to(device)
    return model


if __name__ == "__main__":
    while True:
        data = get_kehu_data()
        fe = get_kehu_fe()
        # 加载模型
        model = get_model_LSTM()
        final_data = predict_future(model, data, 100)
        # model = get_model_Transformer_Duo()
        # final_data = predict_future_duo(model, data, fe, 100)

        # print(final_data[-100:])
        print(scaler.inverse_transform(final_data[-100:].reshape(-1, 1)))

        pyplot.plot(scaler.inverse_transform(final_data[-100:].reshape(-1, 1)))
        pyplot.plot(truth_data)
        pyplot.show()
        break
