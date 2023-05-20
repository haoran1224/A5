import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import sys

sys.path.append("D:\pyhtonProject\A5\任务123")
import trying

# 此时使用它即可实现是否进行登录，单用户
flag = st.session_state['authentication_status']

if not flag:
    st.warning("请登录!")
else:
    ssss = 0
    st.sidebar.markdown("# 高价值客户预测 🔲️")

    st.title("高价值客户预测 🔲️")
    st.markdown("""这是电力客户价值预测模块，在此部分我们采用了前面两部分的模型，对您输入的数据进行了短期的预测
    并帮您筛选出了在下面的一段时间里，电量消耗最大，即最具有价值的客户群体id""")

    # 进行客户需要输入的数据说明
    with st.expander(
            "这里给您提供了一份输入数据的标准样本，我们希望您的输入的数据可以将电力客户id，日期，缴费水平按照所给的标签进行标注 -> "
            "可以点击这个按钮去扩展这个部分来看一下我们给出的例子 👉 "
    ):
        image = Image.open('D:\pyhtonProject\A5\All_data\Electricity_classification.png')
        st.image(image)

    # 传入客户传入的数据
    data1 = st.file_uploader(
        "请在下方点击输入文件"
        "😇👇",
    )

    if data1 is not None:
        dataframe = pd.read_csv(data1)
        final_data_MAX, total_forecast_data = trying.get_MAX_customer(dataframe, 20)

        st.write(final_data_MAX)

        st.write('这里展示了所有的预测数据，以及部分历史数据')

        st.write(total_forecast_data.iloc[:, -20:])

        st.markdown("""You can download all data""")


        @st.cache_data
        def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return pd.DataFrame(df).to_csv().encode('utf-8')


        csv = convert_df(total_forecast_data)
        st.download_button(
            label="Download all data as CSV",
            data=csv,
            file_name='高价值客户预测结果.csv',
            mime='text/csv',
        )
