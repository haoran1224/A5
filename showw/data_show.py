import pandas
import streamlit as st
import pandas as pd
import numpy as np

import streamlit as st
from PIL import Image


def main():
    # st.set_page_config(layout="wide")  # 设置屏幕展开方式，宽屏模式布局更好
    # st.sidebar.write('文档管理导航栏')
    #
    # add_selectbox = st.sidebar.radio(
    #     "文档管理",
    #     ("上传文档", "下载文档", "文档查询")
    # )

    st.title("电力客户行为分析平台")
    st.header("操作说明：")

    st.subheader("1.电力客户聚类")
    st.markdown("我们将会根据您输入的数据信息，对客户进行分类，您需要根据模块中的提示"
                "进行文件一些修改")
    with st.expander(
            "下面是我们运行后的结果的一份展示图 您可以点击按钮查看👉 "
    ):
        image = Image.open('F:\毕设\截图\分类.png')
        st.image(image)

    st.subheader("2.电力负荷预测")
    st.markdown("我们会根据您给出历史负荷数据，对未来一段时间内的电力消耗进行预测，您"
                "需要根据模块中的提示进行文件一些修改")
    with st.expander(
            "下面是我们运行后的结果的比较图以及模型的基本框架 您可以点击按钮查看👉 "
    ):
        image1 = Image.open('F:\毕设\截图\other\结果对比图.png')
        st.image(image1)

        image2 = Image.open('F:\毕设\\3.中期答辩\图2.png')
        st.image(image2)


    st.subheader("3.高价值客户预测")
    st.markdown("我们会根据您给出的数据，对所有用户进行未来用电量的预测，并给出预测时间内下的高价值客户"
                "您也需要根据模块中的提示进行文件一些修改")
    with st.expander(
            "下面是我们运行后的结果的一份展示图 您可以点击按钮查看👉 "
    ):
        image = Image.open('F:\毕设\截图\\transformer\客户预测.png')
        st.image(image)

    # col1, col2 = st.columns([8, 1])
    # data = np.random.randn(10, 1)
    #
    # col1.subheader("A wider column")
    # col1.line_chart(data)
    #
    # col2.subheader("A narrow column with the data")
    # col2.write(data)


if __name__ == "__main__":
    main()
