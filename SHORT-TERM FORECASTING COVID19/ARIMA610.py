# -*- coding: utf-8 -*-
"""

@author: serdarhelli
"""
from statsmodels.tsa.arima_model import ARIMA
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df1=pd.read_csv("covid_19_turkey.csv")
data=np.asarray(df1['total_cases'])

data=np.reshape(data,(len(data),1))

initial_day=15
value=15
data_train=np.copy(data[0+initial_day:30+initial_day,0])

model = ARIMA(data_train, order=(6,1,0))
model_fit = model.fit(disp=0)
output = model_fit.forecast(steps=value)

    
result=output[0]
real=np.copy(data[45:60,0])


accuracy=np.zeros(value) 
for j in range (0,value):
    accuracy[j]=((abs(real[j]-result[j]))/real[j])*100
    
accuracy_sum=np.sum(accuracy)/value
a = pd.date_range(start='24-APRIL-2020', end='8-MAY-2020')
plt.figure(figsize=(9,5))
plt.title('Covid-19 Modelling(ARIMA)')
plt.xlabel('days ')
plt.ylabel('cases')
plt.plot(a,real,'b-',marker='o',label='Real')
plt.plot(a,result,'g-',marker='*', label='Result')
plt.grid()
plt.legend()