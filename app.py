import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
import streamlit as st



name = add_selectbox = st.sidebar.selectbox(
"COMPANY_NAME",
('IBM','PLC','reliance','GreenMotors'))
input_name = str(name)+'.csv'
st.title('Stock Trend Prediction')
dataset_train=pd.read_csv(input_name)
training_set=dataset_train.iloc[:, 5:6].values

st.subheader('Data from last 20 Days')
st.write(dataset_train.describe())

#visualization
st.subheader('Closing price vs Time chart')
fig = plt.figure(figsize=(12,6))
plt.plot(dataset_train.adjusted_close)
st.pyplot(fig)

st.subheader('Closing price vs Time chart with 10MA')
ma10 = dataset_train.adjusted_close.rolling(10).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma10)
plt.plot(dataset_train.adjusted_close)
st.pyplot(fig)

st.subheader('Closing price vs Time chart with 10MA and 20MA')
ma20 = dataset_train.adjusted_close.rolling(20).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma10)
plt.plot(ma20)
plt.plot(dataset_train.adjusted_close)
st.pyplot(fig)

from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler(feature_range=(0, 1))
training_set_scaled=sc.fit_transform(training_set)


#load my model
model = load_model('stock_price_mode.h5')

#testing
dataset_test = dataset_train[21::-1]
real_stock_price=dataset_test.iloc[:, 5:6].values

dataset_total=pd.concat((dataset_train['adjusted_close'], dataset_test['adjusted_close']), axis=0)
inputs=dataset_total[len(dataset_total)-len(dataset_test)-20:].values
inputs=inputs.reshape(-1, 1)
inputs=sc.transform(inputs)
x_test=[]
for i in range(20, 43):
  x_test.append(inputs[i-20:i, 0])
x_test=np.array(x_test)
x_test=np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
predicted_stock_price=model.predict(x_test)
predicted_stock_price=sc.inverse_transform(predicted_stock_price)

#final graph
st.subheader('predictions vs original')
fig2 = plt.figure(figsize = (12,6))
plt.plot(real_stock_price,'b',label = 'Real Stock Price')
plt.plot(predicted_stock_price,'r',label = 'Predicted Stock Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)

st.subheader('Predicted_value')
user_input = st.number_input('Enter The Input:')
st.write('present day(predicted price):',predicted_stock_price[int(user_input)][0])
st.write('previous day(Real price):',real_stock_price[int(user_input-1)][0])