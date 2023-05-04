import math
import time

import joblib
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
import os
import sys
from sklearn.preprocessing import MinMaxScaler
sys.path.append("D:\pyhtonProject\A5\任务4")
import 任务4.Data_manager


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


def train(train_data, train_data_fe):
    model.train()  # Turn on the train mode
    total_loss = 0.
    start_time = time.time()

    for batch, i in enumerate(range(0, len(train_data) - 1 - 5, batch_size)):
        # 获取电力数据和相关的特征
        data, targets = 任务4.Data_manager.get_batch(train_data, i, batch_size)
        # fe = get_batch_tezhen(train_data_fe, i, batch_size)
        # data = torch.cat((fe, data), dim=2)

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
            data, target = 任务4.Data_manager.get_batch(data_source, i, 1)
            # fe = get_batch_tezhen(data_feature, i, 1)
            # # 拼接数据
            # data = torch.cat((fe, data), dim=2)
            # look like the model returns static values for the output window
            output = eval_model(data, 1)
            if calculate_loss_over_all_values:
                total_loss += criterion(output, target).item()
            else:
                total_loss += criterion(output[-output_window:], target[-output_window:]).item()

            test_result = torch.cat((test_result, output[-1].view(-1).cpu()), 0)
            truth = torch.cat((truth, target[-1].view(-1).cpu()), 0)

    # test_result = test_result.cpu().numpy()
    len(test_result)
    # 预测结果
    pyplot.plot(test_result[:100], color="red")
    # 实际结果
    pyplot.plot(truth[:100], color="blue")
    # 差值
    # pyplot.plot(test_result - truth, color="green")
    pyplot.grid(True, which='both')
    pyplot.axhline(y=0, color='k')
    # pyplot.savefig('../image2/australia_tezhen/transformer-epoch%d.png' % epoch)
    pyplot.close()
    if epoch == 100:
        global total_num
        total_num.append(任务4.Data_manager.scaler.inverse_transform(test_result[:100].reshape(-1, 1)).ravel().tolist())
    return total_loss / i


# 一次1000步，计算误差
def evaluate(eval_model, data_source, data_feature):
    eval_model.eval()  # Turn on the evaluation mode
    total_loss = 0.
    eval_batch_size = 10
    print(len(data_source))
    with torch.no_grad():
        for i in range(0, len(data_source) - 1 - 95, eval_batch_size):
            data, targets = 任务4.Data_manager.get_batch(data_source, i, eval_batch_size)
            # fe = get_batch_tezhen(data_feature, i, eval_batch_size)
            # data = torch.cat((fe, data), dim=2)

            output = eval_model(data, eval_batch_size)
            if calculate_loss_over_all_values:
                total_loss += len(data[0]) * criterion(output, targets).cpu().item()
            else:
                total_loss += len(data[0]) * criterion(output[-output_window:], targets[-output_window:]).cpu().item()
    return total_loss / len(data_source)


train_data, val_data, train_data_fe, val_data_fe = 任务4.Data_manager.get_data()
# 输入数据的维度，节点数，多少层，输出维度
model = LSTM(1, 64, 2, 1).to(device)

criterion = nn.MSELoss()
lr = 0.001
# optimizer = torch.optim.SGD(model.parameters(), lr=lr)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.98)
epochs = 100

train_loss = []
test_loss = []

total_num = []

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

        df_data = pd.DataFrame(total_num)
        df_data.to_excel('../Final_Image/datayuce/LSTM.xlsx')
        # 保存模型
        # torch.save(model.state_dict(), '../save/LSTM_model.pt')
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
