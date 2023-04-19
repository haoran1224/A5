import streamlit as st
import pandas as pd
import numpy as np
import tslearn.metrics as metrics
import matplotlib.pyplot as plt
from tslearn.clustering import TimeSeriesKMeans

from electricity_classification.util import get_data, jiangwei

means = []


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
