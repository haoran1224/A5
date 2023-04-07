import math
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from matplotlib import pyplot
from scipy import stats
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

import warnings
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_window = 100
output_window = 5
batch_size = 10  # batch size

calculate_loss_over_all_values = False


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    feature = read_csv('../data/temp3.csv',
                       usecols=['temperature', 'apparentTemperature', 'dewPoint', 'month', 'hour', 'is_holiday'])
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


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        # pe.requires_grad = False
        self.register_buffer('pe', pe)

    def forward(self, x, fe):
        # print(x.size())
        # print(fe.size())
        return x + self.pe[:x.size(0), :] + fe


class TransAm(nn.Module):
    def __init__(self, feature_size=16, num_layers=2, dropout=0.1):
        super(TransAm, self).__init__()
        self.model_type = 'Transformer'

        self.embedding = nn.Linear(6, feature_size)
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(feature_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=8, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(feature_size, 1)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, fe):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        fe = self.embedding(fe)
        src = self.pos_encoder(src, fe)
        output = self.transformer_encoder(src, self.src_mask)  # , self.src_mask)
        output = self.decoder(output)
        return output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


def train(train_data, train_data_fe):
    model.train()  # Turn on the train mode
    total_loss = 0.
    start_time = time.time()

    for batch, i in enumerate(range(0, len(train_data) - 1, batch_size)):
        # 获取电力数据和相关的特征
        data, targets = get_batch(train_data, i, batch_size)
        fe = get_batch_tezhen(train_data_fe, i, batch_size)

        optimizer.zero_grad()
        output = model(data, fe)

        if calculate_loss_over_all_values:
            loss = criterion(output, targets)
        else:
            loss = criterion(output[-output_window:], targets[-output_window:])

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        log_interval = int(len(train_data) / batch_size / 5)
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
        for i in range(0, len(data_source) - 1):
            # 获取数据
            data, target = get_batch(data_source, i, 1)
            fe = get_batch_tezhen(data_feature, i, 1)
            # look like the model returns static values for the output window
            output = eval_model(data, fe)
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
    pyplot.plot(scaler.inverse_transform(test_result[:1000].reshape(-1, 1)), color="red")
    # 实际结果
    pyplot.plot(scaler.inverse_transform(truth[:500].reshape(-1, 1)), color="blue")
    # 差值
    # pyplot.plot(test_result - truth, color="green")
    pyplot.grid(True, which='both')
    pyplot.axhline(y=0, color='k')
    pyplot.savefig('../image/transformer-epoch%d.png' % epoch)
    pyplot.close()

    return total_loss / i


# 预测存在问题
def predict_future(eval_model, data_source, steps):
    eval_model.eval()
    total_loss = 0.
    test_result = torch.Tensor(0)
    truth = torch.Tensor(0)
    _, data = get_batch(data_source, 0, 1)
    with torch.no_grad():
        for i in range(0, steps, 1):
            input = torch.clone(data[-input_window:])
            input[-output_window:] = 0
            output = eval_model(data[-input_window:])
            data = torch.cat((data, output[-1:]))

    data = data.cpu().view(-1)

    pyplot.plot(data, color="red")
    pyplot.plot(data[:input_window], color="blue")
    pyplot.grid(True, which='both')
    pyplot.axhline(y=0, color='k')
    pyplot.savefig('../image/transformer-future%d.png' % steps)
    pyplot.close()


# 一次1000步，计算误差
def evaluate(eval_model, data_source, data_feature):
    eval_model.eval()  # Turn on the evaluation mode
    total_loss = 0.
    eval_batch_size = 1000
    with torch.no_grad():
        for i in range(0, len(data_source) - 1, eval_batch_size):
            data, targets = get_batch(data_source, i, eval_batch_size)
            fe = get_batch_tezhen(data_feature, i, eval_batch_size)
            output = eval_model(data, fe)
            if calculate_loss_over_all_values:
                total_loss += len(data[0]) * criterion(output, targets).cpu().item()
            else:
                total_loss += len(data[0]) * criterion(output[-output_window:], targets[-output_window:]).cpu().item()
    return total_loss / len(data_source)


scaler = MinMaxScaler(feature_range=(-1, 1))
train_data, val_data, train_data_fe, val_data_fe = get_data()
model = TransAm().to(device)

criterion = nn.MSELoss()
lr = 0.001
# optimizer = torch.optim.SGD(model.parameters(), lr=lr)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
# 调整学习率机制
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.98)

best_val_loss = float("inf")
epochs = 150
# The number of epochs
best_model = None

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
                # predict_future(model, val_data, 200)
            else:
                val_loss = evaluate(model, val_data, val_data_fe)
                train_data_loss = evaluate(model, train_data, train_data_fe)

                train_loss.append(train_data_loss)
                test_loss.append(val_loss)

            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.5f} | valid ppl {:8.2f}'.format(epoch, (
                    time.time() - epoch_start_time),
                                                                                                          val_loss,
                                                                                                          math.exp(
                                                                                                              val_loss)))
            print('-' * 89)
            scheduler.step()

        print('******************')
        pyplot.plot(train_loss, color="red")
        # 实际结果
        pyplot.plot(test_loss, color="blue")
        pyplot.grid(True, which='both')
        pyplot.axhline(y=0, color='k')
        pyplot.show()
        break


# 相关测试
def testtest1():
    src, tgt = get_data()
    input1, target = get_batch(src, 0, 10)
    # [100,10,1]
    print(input1)
    pos_encoder = PositionalEncoding(250)
    input1 = pos_encoder(input1)
    print(input1.shape)
    print('**********************')
    from pandas import read_csv
    series = read_csv('../data/temp3.csv', usecols=['gen', 'month'], )
    # 200,2
    series = series[:200]
    series = np.array(series)
    # 200,2
    print(series.shape)
    temp = creat_inout_sq(series, input_window)
    # 100,100,2
    print(temp.shape)
    temp = temp[0:10]
    # 10,100,2
    print(temp.shape)
    # 10,100,1,2
    input = torch.stack(torch.chunk(temp, input_window, 1))
    print(input.shape)
    # 10,100,2 进行降维
    input = torch.squeeze(input, 2)
    print(input.shape)
