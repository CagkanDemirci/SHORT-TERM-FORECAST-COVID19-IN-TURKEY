# -*- coding: utf-8 -*-
"""


@author: serdarhelli
"""

#####U2 SCHEMA#####

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
import tensorflow.keras.losses
import matplotlib.pyplot as plt
import pandas as pd


tf.random.set_seed(1234)
#data preparing
data=np.loadtxt("/Covid19.txt")
difference=np.loadtxt("/Covid19.txt")

#prediction of numbers days=value
value=15
#for training initial day==> until initial_day-initial_day+30 days===>x_train
# until initial_day+1-initial_day+31 days ===>y_train
initial_day=14
data=data[initial_day:initial_day+31+value,1]
#for prediction, data_train [31:46,0] must be zero
data=np.reshape(data,(len(data),1))
data_train=np.zeros([31+value,1])
data_train[0:31,0]=data[0:31,0]
 #normalizng   
scaler = MinMaxScaler(feature_range=(0, 1))
data_train = scaler.fit_transform(data_train)
result=np.zeros([value])
#model loop
for ival in range (0,value):  
    #x_train, y_train in loops, output is new input
    x_train=np.copy(data_train[ival:30+ival,0])
    y_train=np.copy(data_train[ival+1:31+ival,0])
    #model
    model = tf.keras.Sequential()
    #for first 30 days activation tanh, others days activation elu 
    model.add(layers.LSTM(32,activation='elu',input_shape=(30,1),return_sequences=True))

    model.add(layers.Dense(1))
    model.compile(optimizer='adam',loss=tf.keras.losses.MeanSquaredError(),metrics =["accuracy"])
    model.summary()  
    #reshape
    x_train = np.reshape(x_train, (1,30,1))
    y_train =np.reshape(y_train, (1,30,1))
    #fit
    model.fit(x_train,y_train,batch_size=1,epochs=3000,verbose=1)
    #predict
    prediction=model.predict(y_train)
    #for looking results
    result[ival]=prediction[:,-1,:]
    #output is new input
    data_train[31+ival,0]=prediction[:,-1,:]
    model.reset_states()
result=np.reshape(result,(len(result),1))
result = scaler.inverse_transform(result)



real=np.copy(difference[initial_day+31:initial_day+31+value,1])
real=np.reshape(real,(len(real),1))


#error or accuracy       
accuracy=np.zeros(value) 
for j in range (0,value):
    accuracy[j]=((abs(real[j]-result[j]))/real[j])*100
    
accuracy_sum=np.sum(accuracy)/value

a = pd.date_range(start='24-APRIL-2020', end='8-MAY-2020')
plt.figure(figsize=(9,5))
plt.title('Covid-19 Modelling(u2)')
plt.xlabel('days ')
plt.ylabel('cases')
plt.plot(a,real,'b-',marker='o',label='Real')
plt.plot(a,result,'g-',marker='*', label='Result')


