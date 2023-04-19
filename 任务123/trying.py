import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv


def data_arrange():
    data = read_csv("Tianchi_power.csv")

    print(data.isnull().any())  # 没有缺失值
    df = pd.DataFrame(data, columns=['user_id', 'record_date', 'power_consumption'])
    print(df.head())

    # 改变数据类型
    df.loc[:, 'power_consumption'] = df.power_consumption.astype(float)
    df.loc[:, 'user_id'] = df['user_id'].astype(int)
    df.loc[:, 'record_date'] = pd.to_datetime(df.record_date)

    # 按照id和日期进行重新排序
    df = df.sort_values(['user_id', 'record_date'], ascending=[True, True])
    df = df.reset_index(drop=True)
    print(df.head(20))

    # 按照日期和时间绘制数据透视表，获得不同时间下的用户用电数据
    df0 = pd.pivot_table(data=df, columns=['record_date'], values='power_consumption', index=['user_id'])

    print(df0.head(10))

    # df0.to_csv("Tianchi_power_finish.csv")


if __name__ == "__main__":
    while True:
        data = read_csv("Tianchi_power_finish.csv")
        data = np.array(data)
        # 在这里进行调入用户曲线分类，电量使用预测，其他维度进行价值划分
        print(data[0][-100:])
        break
