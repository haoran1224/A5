import streamlit as st
import pandas as pd
import numpy as np
import tslearn.metrics as metrics
import matplotlib.pyplot as plt
from lttb import lttb
from tslearn.clustering import TimeSeriesKMeans

from electricity_classification.util import get_data, jiangwei

means = []

# 读取数据
def get_data(data):
    # data = read_csv("2015.01.01—2016.08.31中国江苏省扬中市1000多家企业每日用电量数据.csv")
    df = pd.DataFrame(data, columns=['user_id', 'record_date', 'power_consumption'])
    print(df.head())

    # 查看数据量
    print("当前数据集含有%s行,%s列" % (df.shape[0], df.shape[1]))

    # 查看id，可知不同的用户共有1454户
    print(df['user_id'].value_counts())

    # 改变数据类型
    df.loc[:, 'power_consumption'] = df.power_consumption.astype(float)
    df.loc[:, 'user_id'] = df['user_id'].astype(int)
    df.loc[:, 'record_date'] = pd.to_datetime(df.record_date)

    # 上述数据的时间仅为一列，无法较好的反映工作日与周末及每月具体日期的区别，因此尝试添加列进一步细化日期
    # 添加一代表星期的列，isoweekday会根据日期判定是周几
    df.loc[:, 'type_day'] = df.record_date.apply(lambda x: x.isoweekday())

    # 添加一代表日期的列，day会根据具体日期判定是几号
    df.loc[:, 'day_of_month'] = df.record_date.apply(lambda x: x.day)

    # 按照id和日期进行重新排序
    df = df.sort_values(['user_id', 'record_date'], ascending=[True, True])
    df = df.reset_index(drop=True)
    print(df.head())

    # 筛选出工作日
    df0 = df[df.type_day <= 5]

    # 按照日期和时间绘制数据透视表，获得不同时间下的用户用电数据
    df0 = pd.pivot_table(data=df, columns=['record_date'], values='power_consumption', index=['user_id'])
    print(df0.head())

    # 用户用电曲线
    # fig, ax = plt.subplots(figsize=(16, 9))
    # plt.plot(df0.T, alpha=1, lw=1)
    # plt.ylabel('KW')
    # plt.show()
    return df0


# total (1454,609)----->(1454,50)进行了数据的降维
def jiangwei(data):
    # 数组标记
    # 一行一共有多少数据
    tre = np.array([range(data.shape[1])])

    # 共有多少组数据
    num = data.shape[0]
    # 声明一个空数组
    total = np.empty((num, 50))

    for j in range(num):
        temp = np.dstack((tre, data[j]))
        # 将第一步的一维降维
        temp = np.squeeze(temp)
        small_data = lttb.downsample(temp, n_out=50)
        total[j] = small_data[:, 1]
    print('降维结束')
    return total

def label_predict(user_data):
    # 数据预处理
    user_data = get_data(user_data)
    user_data_jiangwei = jiangwei(user_data)
    temp_data = metrics.cdist_dtw(user_data_jiangwei)
    # 模型加载，进行分类
    user_KMeans = TimeSeriesKMeans().from_json('../../electricity_classification/try.json')
    group = user_KMeans.predict(temp_data)
    # 显示共有多少户
    num = data.index
    cls = pd.DataFrame(list(num))
    cls['cluster'] = list(group)
    cls.columns = ['user_id', 'cluster']
    # 保存已经进行分类的文件
    cls.to_csv('./solved_data/solved111.csv')
    # 通过排序可以得到每个类中的用户id
    cls = cls.sort_values(by='cluster', ascending=True)
    cls.reset_index(drop=True)
    for i in range(3):
        print(np.array(cls.loc[cls.cluster == i].user_id))
    return cls


def plot_AllData(data):
    # 用户用电曲线
    fig, ax = plt.subplots(figsize=(16, 9))
    plt.plot(data.T, alpha=1, lw=1)
    plt.ylabel('KW')
    plt.show()


def plot_three(data, predictions):
    fig, ax = plt.subplots(figsize=(16, 9))
    for i in range(0, 3):
        all_data = []
        for x, y in zip(data, predictions):
            if y == i:
                all_data.append(x)
                plt.subplot(3, 1, i + 1)
                plt.plot(x, alpha=0.06, color="blue", lw=2)
                plt.xlim(0, 96)
                plt.title('Cluster%s' % (i + 1))
                plt.ylabel('用电量/kW')

        all_data_array = np.array(all_data)
        mean = all_data_array.mean(axis=0)
        means.append(mean)
        plt.plot(mean, color="black", linewidth=4)
    plt.show()


def plot_AllThree():
    """Plots the mean of each cluster in single plot"""
    fig, ax = plt.subplots(figsize=(16, 9))

    for i, item in enumerate(means):
        plt.plot(item, label="cluster %s" % (str(i + 1)))
        plt.xlim(0, 96)
    plt.ylabel('用电量/kW')
    plt.xticks([0, 20, 40, 60, 80, 100, 120, 140],
               ['2015-1-1', '2015-3-31', '2015-6-30', '2015-9-30', '2015-12-31', '2016-3-31', '2016-6-30',
                '2016-8-31'], rotation=60)
    plt.grid()
    plt.legend()
    plt.show()


# 此时使用它即可实现是否进行登录，单用户
flag = st.session_state['authentication_status']

if not flag:
    st.warning("Please login!")
else:
    st.sidebar.markdown("# Electricity classification 2 ❄️")

    st.title("Electricity classification 2 ❄️")
    st.markdown("""这是电力分类模块，在此部分我们采用了K-shape，LTTB降维的方法,通过对电力曲线形状的分类来进行
    判断您所输入的数据中各类用户所占的百分比，以及分析""")

    row1_col1, row1_col2 = st.columns([2, 1])

    with row1_col1:
        # 进行客户需要输入的数据说明
        with st.expander(
                "Steps: Draw a rectangle on the map -> Export it as a GeoJSON -> Upload it back to the app -> Click "
                "the Submit button. Expand this tab to see a demo 👉 "
        ):
            video_empty = st.empty()
        # 传入客户传入的数据
        data = st.file_uploader(
            "Upload a GeoJSON file to use as an ROI. Customize timelapse parameters and then click the Submit button "
            "😇👇",
        )

        if data is not None:
            # Can be used wherever a "file-like" object is accepted:
            dataframe = pd.read_csv(data)
            st.write(dataframe)
            new_dataframe = label_predict(dataframe)
            ssss = 1

    with row1_col2:

        st.stable(new_dataframe)

        @st.cache
        def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')


        csv = convert_df(new_dataframe)

        st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name='large_df.csv',
            mime='text/csv',
        )
