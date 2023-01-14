import numpy as np
import pickle
import streamlit as st
from PIL import Image

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

# 标题
st.title('Flower Class Prediction')

# 页面分左右两列，列宽比例3:2
col1, col2 = st.columns([3, 2])

# 第一列为输入
with col1:
    st.header('Input')
    with st.form('my_form'):
        sepal_length = st.number_input('Input the sepal length')
        sepal_width = st.number_input('Input the sepal width')
        petal_length = st.number_input('Input the petal length')
        petal_width = st.number_input('Input the petal_width')
        submitted = st.form_submit_button('Submit')

# python代码
features = [[sepal_length, sepal_width, petal_length, petal_width]]
model = pickle.load(open("model.pkl", "rb"))
prediction = model.predict(features)[0]
mapping = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
results = mapping[prediction]

# 第二列为输出
with col2:
    st.header('Otput')
    if submitted and results == 'setosa':
        st.subheader(f'The predicted flower species is `{results}`!')
        st.image(Image.open('setosa.png'))
    elif submitted and results == 'versicolor':
        st.subheader(f'The predicted flower species is `{results}`!')
        st.image(Image.open('versicolor.png'))
    elif submitted and results == 'virginica':
        st.subheader(f'The predicted flower species is `{results}`!')
        st.image(Image.open('virginica.png'))
    else:
        st.subheader('Place your input the `parameters` about :hibiscus:!')
