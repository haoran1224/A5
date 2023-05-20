import pandas as pd
import numpy as np
from pandas import read_csv
import sys

sys.path.append("D:\pyhtonProject\A5\任务4")
import forecast_main


def data_arrange(data):
    # data = read_csv('Tianchi_power.csv')
    print(data.isnull().any())  # 没有缺失值
    df = pd.DataFrame(data, columns=['user_id', 'record_date', 'power_consumption'])
    # print(df.head())

    # 改变数据类型
    df.loc[:, 'power_consumption'] = df.power_consumption.astype(float)
    df.loc[:, 'user_id'] = df['user_id'].astype(int)
    df.loc[:, 'record_date'] = pd.to_datetime(df.record_date)

    # 按照id和日期进行重新排序
    df = df.sort_values(['user_id', 'record_date'], ascending=[True, True])
    # 重置索引，并将一开始的索引删除
    df = df.reset_index(drop=True)
    # print(df.head(20))

    # 按照日期和时间绘制数据透视表，获得不同时间下的用户用电数据
    df0 = pd.pivot_table(data=df, columns=['record_date'], values='power_consumption', index=['user_id'])

    # print(df0.head(10))

    # df0.to_csv("Tianchi_power_finish.csv")
    return df0


def data_forecast(data, number):
    # data = read_csv("Tianchi_power_finish.csv")

    # 生成index
    last_date = np.datetime64(data.columns[-1], 'D')
    index_date = np.arange(last_date, last_date + 100)
    empty_date = []
    for temp_data in index_date:
        empty_date.append(str(temp_data)[:10])
    index_date = np.array(empty_date)

    arr = np.arange(number)
    # 创建一个空的用于拼接
    df = pd.DataFrame()

    for i in range(number):
        temp_data = data.iloc[i, 1:]
        final_data = forecast_main.customer_forecast(temp_data, 5)
        dataframe = pd.DataFrame(final_data[-100:])
        df = pd.concat([df, dataframe], axis=1)
    # 设置index
    df.columns = arr
    df.index = index_date
    # print(df.T)
    return df.T


def get_max(data, number):
    # 创建一个空的用于拼接
    dataframe = pd.DataFrame()

    for i in range(number, 0, -1):
        # 取出数据
        data_need = data.iloc[:, -i]
        column_name = np.datetime64(data.columns[-i], 'D')
        print(column_name)
        # 排序
        df = data_need.sort_values(ascending=False)
        df = df.reset_index()
        df = df.iloc[0:20, :]
        df.columns = [f'{column_name}/id', f'consumption{6 - i}/KW']
        # 拼接数据
        dataframe = pd.concat([dataframe, df], axis=1)

    print(dataframe)
    return dataframe


# 用于外部接口
def get_MAX_customer(data, number):
    # 对数据进行处理，生成客户，
    data = data_arrange(data)
    # 对传入的客户数据进行预测,此时传入的是数据的长度
    data = data_forecast(data, data.shape[0])
    # 筛选出预测时间段下的最大用电量用户，此时传入的应是预测步长
    final_data = get_max(data, 5)

    return final_data, data


# 用于后部测试
if __name__ == "__main__":
    while True:
        # data = data_arrange()
        data = data_forecast(20)

        get_max(data, 2)
        break
