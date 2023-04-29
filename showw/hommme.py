import streamlit as st
import streamlit_authenticator as stauth

# 如下代码数据，可以来自数据库
from data_show import main

st.set_page_config(layout="wide")  # 设置屏幕展开方式，宽屏模式布局更好


names = ['肖永威', '管理员']
usernames = ['xiaoyw', 'admin']
passwords = ['S0451', 'ad4516']
hashed_passwords = stauth.Hasher(passwords).generate()
credentials = {"usernames": {}}

for uname, name, pwd in zip(usernames, names, hashed_passwords):
    user_dict = {"name": name, "password": pwd}
    credentials["usernames"].update({uname: user_dict})

authenticator = stauth.Authenticate(credentials,
                                    'some_cookie_name', 'some_signature_key', cookie_expiry_days=30)

name, authentication_status, username = authenticator.login('Login', 'main')


if authentication_status:
    with st.sidebar:
        authenticator.logout('Logout', 'main')
    main()
elif authentication_status == False:
    st.error('Username/password is incorrect')
elif authentication_status == None:
    st.warning('Please enter your username and password')
