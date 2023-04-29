import pandas
import streamlit as st
import pandas as pd
import numpy as np

import streamlit as st


def main():
    # st.set_page_config(layout="wide")  # 设置屏幕展开方式，宽屏模式布局更好
    # st.sidebar.write('文档管理导航栏')
    #
    # add_selectbox = st.sidebar.radio(
    #     "文档管理",
    #     ("上传文档", "下载文档", "文档查询")
    # )

    st.write("操作说明")

    col1, col2 = st.columns([8,1])
    data = np.random.randn(10, 1)

    col1.subheader("A wider column")
    col1.line_chart(data)

    col2.subheader("A narrow column with the data")
    col2.write(data)


if __name__ == "__main__":
    main()
