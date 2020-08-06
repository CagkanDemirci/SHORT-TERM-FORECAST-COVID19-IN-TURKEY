# -*- coding: utf-8 -*-
"""

@author: serdarhelli
"""


from statsmodels.tsa.api import ExponentialSmoothing
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df=pd.read_csv("covid_19_turkey.csv")

data=np.asarray(df['total_cases'])

data=np.reshape(data,(len(data),1))

initial_day=15
value=15
data_train=np.copy(data[0+initial_day:30+initial_day,0])

fit = ExponentialSmoothing(data_train, trend='add' ,damped=True).fit(damping_slope=0.96,optimized=True)
fcast = fit.forecast(value)
result=np.copy(fcast)



real=np.copy(data[30+initial_day:30+initial_day+value,0])
a = pd.date_range(start='24-APRIL-2020', end='8-MAY-2020')
accuracy=np.zeros(value) 
for j in range (0,value):
    accuracy[j]=((abs(real[j]-result[j]))/real[j])*100
plt.figure(figsize=(9,5))
plt.title('Covid-19 Modelling(Holts Method(Damped=True))')
plt.xlabel('days ')
plt.ylabel('cases')
plt.plot(a,real,'b-',marker='o',label='Real')
plt.plot(a,result,'g-',marker='*', label='Result')
plt.grid()
plt.legend()

