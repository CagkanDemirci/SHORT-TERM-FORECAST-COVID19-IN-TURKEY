# -*- coding: utf-8 -*-
"""


@author: serdarhelli
"""

import pandas as pd
from fbprophet import Prophet
import numpy as np
import matplotlib.pyplot as plt

df1=pd.read_csv("covid_19_turkey.csv")
data=np.asarray(df1['total_cases'])

data=np.reshape(data,(len(data),1))

initial_day=15
value=15
data_train=np.copy(data[0+initial_day:30+initial_day,0])
days = pd.date_range(start='25-MARCH-2020', end='23-APRIL-2020')
df=pd.DataFrame({'ds':days,'y':data_train})
real=np.copy(data[45:60,0])

my_model = Prophet(interval_width=0.95,daily_seasonality=True)
my_model.fit(df)
future_dates = my_model.make_future_dataframe(periods=1*15, freq='1d')
future_dates.tail()
forecast = my_model.predict(future_dates)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
my_model.plot_components(forecast)
my_model.plot(forecast,uncertainty=True)
trends=np.float64(forecast['trend'])
result=np.copy(trends[29:44])

accuracy=np.zeros(15) 
for j in range (0,15):
    accuracy[j]=((abs(real[j]-result[j]))/real[j])*100
    
accuracy_sum=np.sum(accuracy)/15



