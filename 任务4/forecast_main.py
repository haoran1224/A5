import joblib

# 模型
import torch
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler
from torch import nn
import numpy as np
import os
import pandas as pd
import sys

sys.path.append("D:\pyhtonProject\A5\任务4")
sys.path.append("D:\pyhtonProject\A5\任务4\code")
import Data_manager
import MyTransformer
import MyLSTM

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_window = 100
output_window = 5
scaler = MinMaxScaler(feature_range=(-1, 1))

truth_data = None

t = torch.zeros(output_window)
t = t.view(len(t), 1, -1).to(device)

###############################################预测部分#############################################

# 单维预测
def predict_future_transformer(eval_model, data, steps):
    eval_model.eval()
    with torch.no_grad():
        # 获取最后的95条数据
        input = torch.clone(data[-input_window + output_window:])
        # 拼接成lstm输入的数据
        input_data = torch.cat((input, t))
        # 预测
        output = eval_model(input_data, input_data[-8:].T)
        # 原数据拼接
        data = torch.cat((data, output[-output_window:]))

    data = data.cpu().view(-1)

    pyplot.plot(data[-input_window - steps:], color="red")
    pyplot.plot(data[-input_window - steps:-steps], color="blue")
    pyplot.grid(True, which='both')
    pyplot.axhline(y=0, color='k')
    pyplot.show()
    pyplot.close()

    return data


# 单维预测
def predict_future_lstm(eval_model, data, steps):
    eval_model.eval()
    with torch.no_grad():
        # 获取最后的95条数据
        input = torch.clone(data[-input_window + output_window:])
        # 拼接成lstm输入的数据
        input_data = torch.cat((input, t))
        # 预测
        output = eval_model(input_data, 1)
        # 原数据拼接
        data = torch.cat((data, output[-output_window:]))

    data = data.cpu().view(-1)
    # # 全部数据+预测部分
    # pyplot.plot(data[-input_window:], color="red")
    # # 实际数据
    # pyplot.plot(data[-input_window:-output_window], color="blue")
    # pyplot.grid(True, which='both')
    # pyplot.axhline(y=0, color='k')
    # pyplot.show()
    # pyplot.close()

    return data


# 多维预测
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

    pyplot.plot(data[-input_window - steps:], color="red")
    pyplot.plot(data[-input_window - steps:-steps], color="blue")
    pyplot.grid(True, which='both')
    pyplot.axhline(y=0, color='k')
    pyplot.show()
    # pyplot.savefig('../image/transformer-future%d.png' % steps)
    pyplot.close()

    return data


#############################################数据处理部分##############################################

def get_kehu_data(data):
    # data = pd.read_csv('data/Australia.csv', usecols=['电力负荷'])
    # data = data.iloc['电力负荷']
    data = data.loc[:, '电力负荷']
    # 归一化
    global scaler
    data = scaler.fit_transform(np.array(data).reshape(-1, 1)).reshape(-1)
    data = data[0:2800]

    # 转换为张量
    data = torch.FloatTensor(data)
    # 转换为（100，batch_size,维度）
    data = data.view(len(data), 1, -1).to(device)

    return data

def get_kehu_fe(feature):
    # feature = pd.read_csv('data/Australia.csv', usecols=['month', 'day', '小时', '干球温度', '露点温度', '湿球温度', '湿度', '电价'])
    feature = Data_manager.chuli(feature)
    feature = feature.loc[0:3999, :]
    feature = np.array(feature)
    # 转换为张量
    data = torch.FloatTensor(feature)
    # 转换为（100，batch_size,维度）
    data = data.reshape(data.shape[0], 1, 8).to(device)
    return data

def get_customer_data(data):
    # 归一化
    global scaler
    data = scaler.fit_transform(np.array(data).reshape(-1, 1)).reshape(-1)
    # 转换为张量
    data = torch.FloatTensor(data)
    # 转换为（100，batch_size,维度）
    data = data.view(len(data), 1, -1).to(device)
    return data


###################################################模型部分############################################

# LSTM-单变量
def get_model_LSTM():
    model = MyLSTM.LSTM(1, 64, 2, 1)
    model.load_state_dict(torch.load('D:\pyhtonProject\A5\任务4\save\LSTM_model.pt'))
    model.to(device)
    return model


# TRANSFORMER-单变量
def get_model_Transformer():
    model = MyTransformer.TransAm()
    model.load_state_dict(torch.load('D:\pyhtonProject\A5\任务4\save\TRANSFORMER_model.pt'))
    model.to(device)
    return model


# TRANSFORMER-多变量
def get_model_Transformer_Duo():
    model = MyTransformer.TransAm()
    model.load_state_dict(torch.load('D:\pyhtonProject\A5\任务4\save\TRANSFORMER_model_Duo.pt'))
    model.to(device)
    return model

####################################################外部接口部分#######################################

# 用于外部接口部分，需要对数据进行判断，进行二次调用处理
def total(data, number):
    data = get_kehu_data(data)
    model = get_model_Transformer()
    final_data = predict_future_transformer(model, data, number)
    return scaler.inverse_transform(final_data[-200:].reshape(-1, 1))


def customer_forecast(temp_data, number):
    data = get_customer_data(temp_data)
    model = get_model_LSTM()
    final_data = predict_future_lstm(model,data,number)
    return scaler.inverse_transform(final_data.reshape(-1,1))


if __name__ == "__main__":
    while True:
        data = get_kehu_data()

        # fe = get_kehu_fe()
        # 加载模型
        # model = get_model_LSTM()
        # final_data = predict_future_lstm(model, data, 100)
        # model = get_model_Transformer_Duo()
        # final_data = predict_future_duo(model, data, fe, 100)

        model = get_model_Transformer()
        final_data = predict_future_transformer(model, data, 50)
        # print(final_data[-100:])
        # pyplot.plot(final_data[-100:])

        print(scaler.inverse_transform(final_data[-100:].reshape(-1, 1)))

        break
