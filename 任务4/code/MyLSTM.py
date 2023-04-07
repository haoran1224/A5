import copy
import math
import time

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

calculate_loss_over_all_values = False

input_window = 100
output_window = 5
batch_size = 10  # batch size


# 模型
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.num_directions = 1  # 单向LSTM
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=False)
        self.linear = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input_seq, batch_size):
        # batch_size, seq_len = input_seq.shape[0], input_seq.shape[1]
        # 可以使用其他的代替
        h_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(device)
        c_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(device)
        # output(batch_size, seq_len, num_directions * hidden_size)
        output, _ = self.lstm(input_seq, (h_0, c_0))  # output(5, 30, 64)
        pred = self.linear(output)  # (5, 30, 1)
        # pred = pred[:, -1, :]  # (5, 1)
        return pred


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
    data = read_csv('../data/temp3.csv', usecols=['gen'])
    # 归一化
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    amplitude = scaler.fit_transform(data.to_numpy().reshape(-1, 1)).reshape(-1)
    # amplitude = scaler.fit_transform(amplitude.reshape(-1, 1)).reshape(-1)
    # 划分训练测试集
    sampels = 5800
    train_data = amplitude[:sampels]
    test_data = amplitude[sampels:7000]

    # convert our train data into a pytorch train tensor
    # train_tensor = torch.FloatTensor(train_data).view(-1)
    # todo: add comment..
    train_sequence = create_inout_sequences(train_data, input_window)
    train_sequence = train_sequence[:-output_window]  # todo: fix hack?

    # test_data = torch.FloatTensor(test_data).view(-1)
    test_data = create_inout_sequences(test_data, input_window)
    test_data = test_data[:-output_window]  # todo: fix hack?

    # 读取特征数据 用于拼接
    feature = read_csv('../data/temp3.csv',
                       usecols=['temperature', 'apparentTemperature', 'dewPoint', 'month', 'hour', 'is_holiday'])
    feature = chuli(feature)
    feature = np.array(feature)

    feature_train = feature[:sampels]
    feature_test = feature[sampels:7000]

    feature_train = creat_inout_sq(feature_train, input_window)
    feature_train = feature_train[:-output_window]

    feature_test = creat_inout_sq(feature_test, input_window)
    feature_test = feature_test[:-output_window]

    return train_sequence.to(device), test_data.to(device), feature_train.to(device), feature_test.to(device)


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


def train(train_data, train_data_fe):
    model.train()  # Turn on the train mode
    total_loss = 0.
    start_time = time.time()

    # print(len(train_data))
    # print(batch_size)
    # print('******************')
    for batch, i in enumerate(range(0, len(train_data) - 1 - 5, batch_size)):
        # 获取电力数据和相关的特征
        data, targets = get_batch(train_data, i, batch_size)
        fe = get_batch_tezhen(train_data_fe, i, batch_size)
        data = torch.cat((fe, data), dim=2)

        optimizer.zero_grad()
        output = model(data, batch_size)

        if calculate_loss_over_all_values:
            loss = criterion(output, targets)
        else:
            # print(output.shape)
            # print(targets.shape)
            loss = criterion(output[-output_window:], targets[-output_window:])

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        log_interval = int(len(train_data) / batch_size / 5)
        # print('********************')
        # print(log_interval)
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.6f} | {:5.2f} ms | '
                  'loss {:5.5f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // batch_size, scheduler.get_lr()[0],
                              elapsed * 1000 / log_interval,
                cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()


def plot_and_loss(eval_model, data_source, data_feature, epoch):
    eval_model.eval()
    total_loss = 0.
    test_result = torch.Tensor(0)
    truth = torch.Tensor(0)
    with torch.no_grad():
        for i in range(0, len(data_source) - 1 - 5):
            # 获取数据
            data, target = get_batch(data_source, i, 1)
            fe = get_batch_tezhen(data_feature, i, 1)
            # 拼接数据
            data = torch.cat((fe, data), dim=2)
            # look like the model returns static values for the output window
            output = eval_model(data, 1)
            if calculate_loss_over_all_values:
                total_loss += criterion(output, target).item()
            else:
                total_loss += criterion(output[-output_window:], target[-output_window:]).item()

            test_result = torch.cat((test_result, output[-1].view(-1).cpu()),
                                    0)  # todo: check this. -> looks good to me
            truth = torch.cat((truth, target[-1].view(-1).cpu()), 0)

    # test_result = test_result.cpu().numpy()
    len(test_result)
    # 预测结果
    pyplot.plot(test_result, color="red")
    # 实际结果
    pyplot.plot(truth[:500], color="blue")
    # 差值
    # pyplot.plot(test_result - truth, color="green")
    pyplot.grid(True, which='both')
    pyplot.axhline(y=0, color='k')
    pyplot.savefig('../image2/transformer-epoch%d.png' % epoch)
    pyplot.close()

    return total_loss / i


# 一次1000步，计算误差
def evaluate(eval_model, data_source, data_feature):
    eval_model.eval()  # Turn on the evaluation mode
    total_loss = 0.
    eval_batch_size = 10
    print(len(data_source))
    with torch.no_grad():
        for i in range(0, len(data_source) - 1 - 95, eval_batch_size):
            data, targets = get_batch(data_source, i, eval_batch_size)
            fe = get_batch_tezhen(data_feature, i, eval_batch_size)
            data = torch.cat((fe, data), dim=2)

            output = eval_model(data, eval_batch_size)
            if calculate_loss_over_all_values:
                total_loss += len(data[0]) * criterion(output, targets).cpu().item()
            else:
                total_loss += len(data[0]) * criterion(output[-output_window:], targets[-output_window:]).cpu().item()
    return total_loss / len(data_source)


train_data, val_data, train_data_fe, val_data_fe = get_data()
# 输入数据的维度，节点数，多少层，输出维度
model = LSTM(7, 64, 2, 1).to(device)

criterion = nn.MSELoss()
lr = 0.001
# optimizer = torch.optim.SGD(model.parameters(), lr=lr)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.98)
epochs = 100


train_loss = []
test_loss = []

if __name__ == "__main__":
    while True:
        for epoch in range(1, epochs + 1):
            # 训练数据集
            epoch_start_time = time.time()
            train(train_data, train_data_fe)
            # 测试集评估

            if epoch % 10 == 0:
                val_loss = plot_and_loss(model, val_data, val_data_fe, epoch)
                print('-' * 89)
                print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.5f} | valid ppl {:8.2f}'.format(epoch, (
                        time.time() - epoch_start_time),
                                                                                                              val_loss,
                                                                                                              math.exp(
                                                                                                                  val_loss)))
                print('-' * 89)
            else:
                # 计算误差，看是否过拟合，进行判断是否加多训练轮次
                val_loss_train = evaluate(model, train_data, train_data_fe)
                val_loss_test = evaluate(model, val_data, val_data_fe)
                train_loss.append(val_loss_train)
                test_loss.append(val_loss_test)

        pyplot.plot(train_loss, color="red")
        # 实际结果
        pyplot.plot(test_loss, color="blue")
        pyplot.grid(True, which='both')
        pyplot.axhline(y=0, color='k')
        pyplot.show()
        # # 获取电力数据和相关的特征
        # data, targets = get_batch(train_data, 0, batch_size)
        # fe = get_batch_tezhen(train_data_fe, 0, batch_size)
        # print(data.shape)
        # print(fe.shape)
        # data = torch.cat((fe,data),dim=2)
        # print(data.shape)
        break
