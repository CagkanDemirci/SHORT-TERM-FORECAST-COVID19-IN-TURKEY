# -*- coding: utf-8 -*-
"""


@author: serdarhelli
"""


####U1 SCHEMA####
import pandas as pd

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
import tensorflow.keras.losses
import matplotlib.pyplot as plt
#data
data=np.loadtxt("/Covid19.txt")
difference=np.loadtxt("/Covid19.txt")
#data preparing
value=15
initial_day=14
data=data[initial_day:initial_day+31+value,1]
data=np.reshape(data,(len(data),1))
data_train=np.zeros([31+value,1])
data_train=np.copy(data)
#normalizing
scaler = MinMaxScaler(feature_range=(0, 1))
data_train = scaler.fit_transform(data_train)
result=np.zeros([value])
# loop
for ival in range (0,value):   
#preprocessing
    x_train=np.copy(data_train[ival:30+ival,0])
    y_train=np.copy(data_train[ival+1:31+ival,0])    
    #reshape    
    x_train = np.reshape(x_train, (1,30,1))
    y_train =np.reshape(y_train, (1,30,1))
    #model    
    model = tf.keras.Sequential()
    model.add(layers.LSTM(32,input_shape=(30,1),return_sequences=True))    
    model.add(layers.Dense(1))
    model.compile(optimizer='adam',loss=tf.keras.losses.MeanSquaredError(),metrics =["accuracy"])    
    model.fit(x_train,y_train,batch_size=1,epochs=2000,verbose=1)  
    #predict
    prediction=model.predict(y_train)
    result[ival]=prediction[:,-1,:]         
result=np.reshape(result,(len(result),1))
result = scaler.inverse_transform(result)





real=np.copy(difference[initial_day+31:initial_day+31+value,1])
real=np.reshape(real,(len(real),1))
plt.plot(real)
plt.plot(result)
accuracy=np.zeros(15) 
accuracy_sum=0       

for j in range (0,15):
    accuracy[j]=((abs(real[j]-result[j]))/real[j])*100

accuracy_sum=(np.sum(accuracy))/value

a = pd.date_range(start='24-APRIL-2020', end='8-MAY-2020')

plt.title('Covid-19 Modelling(u1)')
plt.xlabel('days ')
plt.ylabel('cases')
plt.plot(a,real,'b-',marker='o',label='Real')
plt.plot(a,result,'g-',marker='*', label='Result')


plt.legend()