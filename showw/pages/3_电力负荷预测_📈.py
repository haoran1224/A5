import streamlit as st
import pandas as pd

from PIL import Image
import sys
import matplotlib.pyplot as plt

from matplotlib import pyplot

sys.path.append("D:\pyhtonProject\A5\任务4")
import forecast_main


def show(data, input_window, steps):
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.plot(data[-input_window - steps:], color="red", label='future')
    ax.plot(data[-input_window - steps:-steps], color="blue", label='history')
    ax.set_ylabel('Electricity/kW')
    return fig


# 此时使用它即可实现是否进行登录，单用户
flag = st.session_state['authentication_status']

if not flag:
    st.warning("请登录!")
else:
    st.sidebar.markdown("# 电力负荷预测 📈")

    st.title("电力负荷预测 📈")

    st.markdown("""这是电力预测模块，在此部分我们采用了Transformer模型预测方法,可以通过对您的历史负荷数据
    进行分析，预测未来一段时间内您的电力消耗，同时我们会根据您的数据完整程度来自动帮您选择适合的模型进行预测""")

    # 进行客户需要输入的数据说明
    with st.expander(
            "这里给您提供了一份输入数据的标准样本，我们希望您的输入的数据可以将电力消耗数据标签标记为电力负荷 -> "
            "可以点击这个按钮去扩展这个部分来看一下我们给出的例子 👉 "
    ):
        image = Image.open('D:\pyhtonProject\A5\All_data\Electricity_Forecast.png')
        st.image(image)

    # 传入客户传入的数据
    data1 = st.file_uploader(
        "请在下方点击输入文件"
        "😇👇",
    )

    row2_col1, row2_col2 = st.columns([4, 1])
    with row2_col1:
        number = st.number_input('请输入进行预测的步长', value=50)
    with row2_col2:
        button_status = st.button('预测', use_container_width=True)

    if button_status:
        if data1 is not None:
            # Can be used wherever a "file-like" object is accepted:
            dataframe = pd.read_csv(data1)
            st.write(dataframe)

            st.divider()

            final_data = forecast_main.total_duo(dataframe, number)
            row1_col1, row1_col2 = st.columns([2, 1])

            with row1_col1:
                st.markdown("""Future Electricity consumption""")
                st.write(final_data[-number:].T)

            with row1_col2:
                st.markdown("""You can download these data""")


                @st.cache_data
                def convert_df(df):
                    # IMPORTANT: Cache the conversion to prevent computation on every rerun
                    return pd.DataFrame(df).to_csv().encode('utf-8')


                csv = convert_df(final_data[-number:])
                st.download_button(
                    label="Download data as CSV",
                    data=csv,
                    file_name='电力负荷预测结果.csv',
                    mime='text/csv',
                )

            st.divider()

            st.pyplot(show(final_data, 95, number))
        else:
            st.warning('请输入正确格式的数据')
