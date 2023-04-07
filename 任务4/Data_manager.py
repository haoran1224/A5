import MySQLdb
import inline as inline
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

import warnings

warnings.filterwarnings("ignore")


def bianma():
    data = pd.read_excel('data/excel2.xlsx')
    # data = data.drop(['time'], axis=1)
    print(data.head())
    # dumpies = pd.get_dummies(data,columns=['icon','summary'])
    # print(dumpies[:5])
    # dumpies.to_excel('data/excel2.xlsx',index=False)
    data.to_csv('data/temp4.csv', index=False)


# 查看数据分布情况
def look_data():
    data = pd.read_csv('data/temp3.csv', usecols=['gen'])
    train_data = data[:4000]
    test_data = data[3000:4000]

    plt.figure(figsize=(16, 6))
    # sns.boxplot(train_data[:],orient='v',width=0.5)
    plt.plot(train_data)
    plt.show()

    # data = pd.read_excel('data/澳大利亚.xlsx', usecols=['电力负荷'])
    # print(data.info())
    # plt.plot(data.iloc[0:17520])
    # plt.show()


if __name__ == "__main__":
    while True:
        # mysql = MySQLdb.connect(host='localhost', port=3306, user='root', passwd='123456', db='electricity')
        # sql = 'SELECT * from homea2015,homea_meter2015 where homea2015.Time = homea_meter2015.Time'
        # data = pd.read_sql(sql,con=mysql)
        # data.to_excel('data/excel3.xlsx', sheet_name='Sheet1', index=False)
        # bianma()
        look_data()
        break
