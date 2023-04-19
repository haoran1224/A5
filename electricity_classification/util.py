import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
import tslearn.metrics as metrics
import lttb

# 聚类模型
from sklearn.cluster import KMeans
import numpy as np
import lttb
import matplotlib.pyplot as plt
from pandas import read_csv
import pandas as pd
import tslearn.metrics as metrics
from tslearn.clustering import TimeSeriesKMeans


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


# 测试出应该有几个聚类点
def elbow_method_new(temp_data, n_clusters):
    fig, ax = plt.subplots(figsize=(16, 9))
    # 记录不同聚类点下的误差大小
    distortions = []
    for i in range(1, n_clusters):
        km = TimeSeriesKMeans(n_clusters=i,
                              init='k-means++',  # 初始中心簇的获取方式，k-means++一种比较快的收敛的方法
                              n_init=10,  # 初始中心簇的迭代次数
                              max_iter=300,  # 数据分类的迭代次数
                              random_state=0)  # 初始化中心簇的方式
        km.fit(temp_data)
        distortions.append(km.inertia_)  # inertia计算样本点到最近的中心点的距离之和

    plt.plot(range(1, n_clusters), distortions, marker='o', lw=1)
    plt.xlabel('聚类数量')
    plt.ylabel('至中心点距离之和')
    plt.show()


class user_KShape():
    # 模型训练
    def fit(self, n_clusters, temp_data):
        self.n_clusters = n_clusters
        self.kmeans = TimeSeriesKMeans(self.n_clusters)
        self.predictions = self.kmeans.fit_predict(temp_data)
        # 保存模型
        self.kmeans.to_json('../try.json')


# 显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

data = get_data()

data = np.array(data)
# 数据进行降维，在保持原有形状的基础上，进行数据的缩小
data_jiangwei = jiangwei(data)
# 使用dtw计算各个曲线之间的距离，用于提升精度
# temp_data = metrics.cdist_dtw(data_jiangwei)

# elbow_method_new(data_jiangwei, 13)

user_culster = user_KShape()
user_culster.fit(3,data)

my_kemans = TimeSeriesKMeans().from_json('try.json')
pre = my_kemans.predict(data)
print(type(pre))
for i in pre:
    if i == 1:
        print('*********************')
    if i == 2:
        print('#########################')
