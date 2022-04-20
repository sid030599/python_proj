import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader as data
import tensorflow as tf
from keras.models import load_model
import streamlit as st
from sklearn.model_selection import train_test_split
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

start = '2010-01-01'
end = '2021-12-31'

user_input = st.text_input('Enter Stock Ticker','AAPL')
st.title('Stock Trend prediction')
df=data.DataReader(user_input,'yahoo',start,end)

# describing data
st.subheader('Data forom 2010 -2021 ')
st.write(df.describe())

# VISUALIZATION
st.subheader('Closing Price vs time chart:')
fig=plt.figure(figsize=(12,6))
plt.plot(df['Close'])
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 100ma and 200 moving avg')
fig=plt.figure(figsize=(12,6))
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
plt.plot(ma200)
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)

data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
data_training_array= scaler.fit_transform(data_training)
 
#load my model
model=load_model('keras_model.h5')
past_100_days = data_training.tail(100)
final_df =past_100_days.append(data_testing,ignore_index=True)
input_data=scaler.fit_transform(final_df)
x_test=[]
y_test=[]

for i in range(100,input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])

x_test , y_test=np.array(x_test),np.array(y_test)
y_predicted=model.predict(x_test)
scaler.scale_
scale_factor=1/scaler.scale_[0]
y_predicted=y_predicted*scale_factor
y_test=y_test*scale_factor

st.subheader('Prediction vs Origiinal')
fig2=plt.figure(figsize=(12,6))
plt.plot(y_test,'b',label='original Pricee')
plt.plot(y_predicted,'r',label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)