import streamlit as st
import pandas as pd
import numpy as np
import tslearn.metrics as metrics
import matplotlib.pyplot as plt
from lttb import lttb
from tslearn.clustering import TimeSeriesKMeans
from PIL import Image

import sys

sys.path.append("D:\pyhtonProject\A5\electricity_classification")
import util

means = []


def label_predict(user_data):
    # 数据预处理
    user_data = util.get_data(user_data)
    # 转为数组
    # user_data_jiangwei = util.jiangwei(np.array(user_data))
    # temp_data = metrics.cdist_dtw(user_data_jiangwei)
    # 模型加载，进行分类
    user_KMeans = TimeSeriesKMeans().from_json('D:\pyhtonProject\A5\electricity_classification\oory.json')
    group = user_KMeans.predict(np.array(user_data))

    # 显示共有多少户
    num = user_data.index
    cls = pd.DataFrame(list(num))
    cls['cluster'] = list(group)
    cls.columns = ['user_id', 'cluster']

    # 通过排序可以得到每个类中的用户id
    cls = cls.sort_values(by='cluster', ascending=True)
    cls.reset_index(drop=True)
    for i in range(3):
        print(np.array(cls.loc[cls.cluster == i].user_id))
    return cls, user_data, group


def plot_AllData(data):
    # 用户用电曲线
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.plot(data.T, alpha=1, lw=1)
    ax.set_ylabel('KW')
    return fig


def plot_three(data, predictions):
    fig, axes = plt.subplots(3,1, figsize=(16, 9))
    for i in range(0, 3):
        all_data = []
        for x, y in zip(data, predictions):
            if y == i:
                all_data.append(x)
                axes[i].plot(x, alpha=0.06, color="blue", lw=2)
                # plt.xlim(0, 96)
                axes[i].set_title('Cluster%s' % (i + 1))
                axes[i].set_ylabel('用电量/kW')

        all_data_array = np.array(all_data)
        mean = all_data_array.mean(axis=0)
        means.append(mean)
        axes[i].plot(mean, color="black", linewidth=3)

    return fig


def plot_AllThree():
    """Plots the mean of each cluster in single plot"""
    fig, ax = plt.subplots(figsize=(16, 9))

    for i, item in enumerate(means):
        ax.plot(item, label="cluster %s" % (str(i + 1)))

    ax.set_ylabel('用电量/kW')
    ax.set_xticks([0, 90, 180, 270, 360, 450, 540, 630],
               ['2015-1-1', '2015-3-31', '2015-6-30', '2015-9-30', '2015-12-31', '2016-3-31', '2016-6-30',
                '2016-8-31'], rotation=60)
    ax.grid()
    ax.legend()
    return fig


# 此时使用它即可实现是否进行登录，单用户
flag = st.session_state['authentication_status']

if not flag:
    st.warning("Please login!")
else:
    ssss = 0
    st.sidebar.markdown("# Electricity classification 2 ❄️")

    st.title("Electricity classification 2 ❄️")
    st.markdown("""这是电力分类模块，在此部分我们采用了K-shape，LTTB降维的方法,通过对电力曲线形状的分类来进行
    判断您所输入的数据中各类用户所占的百分比，以及分析""")

    row1_col1, row1_col2 = st.columns([2, 1])

    with row1_col1:
        # 进行客户需要输入的数据说明
        with st.expander(
                "这里给出了一份输入数据的标准样本，我们希望您可以给您数据中用户id，日期，电力消耗数据分别标记为user_id,record_date,power_consumption -> "
                "可以点击这个按钮去扩展这个部分来看一下我们给出的例子 👉 "
        ):
            image = Image.open('D:\pyhtonProject\A5\All_data\Electricity_classification.png')
            st.image(image)
        # 传入客户传入的数据
        data = st.file_uploader(
            "Upload a GeoJSON file to use as an ROI. Customize timelapse parameters and then click the Submit button "
            "😇👇",
        )

        if data is not None:
            # Can be used wherever a "file-like" object is accepted:
            dataframe = pd.read_csv(data)
            st.write(dataframe)
            new_dataframe, all_dataframe, group = label_predict(dataframe)
            ssss = 1
            # 进行绘图
            st.pyplot(plot_AllData(all_dataframe))
            st.pyplot(plot_three(np.array(all_dataframe), group))
            # plot_three(np.array(all_dataframe), group)
            st.pyplot(plot_AllThree())

    with row1_col2:
        if ssss == 1:

            @st.cache_data
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
            st.table(new_dataframe)


