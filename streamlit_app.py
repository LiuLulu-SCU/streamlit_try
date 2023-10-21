import numpy as np
import pickle
import streamlit as st
from PIL import Image

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

# 标题
st.title('预测鸢尾花分类')
st.markdown(
    ' **这是一个使用机器学习随机森林算法对鸢尾花数据集进行分类预测的示例，by [刘卢路](https://www.liululu.net/)** ')
st.markdown(
    '请在页面左边输入sepal length、sepal width、petal length和petal width四个参数，并点击`Submit`按钮，即可得到预测的鸢尾花分类！')

# 页面分左右两列，列宽比例3:2
col1, col2 = st.columns([3, 2])

# 第一列为输入
with col1:
    st.header('输入')
    with st.form('my_form'):
        sepal_length = st.number_input('请输入花萼长度')
        sepal_width = st.number_input('请输入花萼宽度')
        petal_length = st.number_input('请输入花瓣长度')
        petal_width = st.number_input('请输入花瓣宽度')
        submitted = st.form_submit_button('提交')

# python代码
features = [[sepal_length, sepal_width, petal_length, petal_width]]
model = pickle.load(open("model.pkl", "rb"))
prediction = model.predict(features)[0]
mapping = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
results = mapping[prediction]

prop =  max(model.predict_proba(features)[0])*100

# 第二列为输出
with col2:
    st.header('输出')
    if submitted and results == 'setosa':
        st.subheader(f'预测的鸢尾花类型为：`{results}`!')
        st.subheader(f'预测概率为{prop}%!')
        st.image(Image.open('setosa.png'))
    elif submitted and results == 'versicolor':
        st.subheader(f'预测的鸢尾花类型为：`{results}`!')
        st.subheader(f'预测概率为{prop}%!')
        st.image(Image.open('versicolor.png'))
    elif submitted and results == 'virginica':
        st.subheader(f'预测的鸢尾花类型为：`{results}`!')
        st.subheader(f'预测概率为{prop}%!')
        st.image(Image.open('virginica.png'))
    else:
        st.subheader('请输入鸢尾花的花萼花瓣参数:hibiscus:!')
    
    
