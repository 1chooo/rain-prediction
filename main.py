# -*- coding: utf-8 -*-
'''
Create Date: 2023/08/24
Author: @1chooo(Hugo ChunHo Lin)
Version: v0.0.2
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import glob
from Plum.Utils.Merge import load_data

df = load_data()
df.drop(['ObsTime', 'SeaPres', 'StnPresMaxTime', 'StnPresMinTime'], axis = 1, inplace = True)
df.drop(['T Max Time', 'T Min Time', 'Td dew point'], axis = 1, inplace = True)
df.drop(['RHMinTime', 'WGustTime'], axis = 1, inplace = True)
df.drop(['PrecpHour', 'PrecpMax10', 'PrecpMax10Time', 'PrecpMax60', 'PrecpMax60Time'], axis = 1, inplace = True)
df.drop(['SunShine', 'SunShineRate', 'GloblRad', 'VisbMean'], axis = 1, inplace = True)
df.drop(['EvapA', 'UVI Max', 'UVI Max Time', 'Cloud Amount'], axis = 1, inplace = True) 
df = df.replace('...','-999')
df = df.replace('/','-999')


for i in range(854):
    for j in range(0, 13):
        if df.iloc[i, j] == '-999':
            df.iloc[i, j] = 0.0

df = pd.DataFrame(df, dtype = np.float64)

for k in range(854):
    if df.iloc[k,12] > 0.0:
        df.iloc[k,12] = 1
    else:
        df.iloc[k,12] = 0

count0, count1, count2, count3, count4, count5, count8 , count10 = 0, 0, 0, 0 ,0, 0, 0, 0
stnprestotal , stnpresmaxtotal , stnpresmintotal, WStotal , WSGusttotal, Ttotal , Tmaxtotal , Tmintotal = 0 , 0 , 0, 0 , 0, 0 , 0 , 0

for k in range(0,853):
    if (df.iloc[k,0] != -999.0):
        stnpres = float(df.iloc[k,0])
        count0 += 1
        stnprestotal += stnpres
    if (df.iloc[k,1] != -999.0):
        stnpresmax = float(df.iloc[k,1])
        count1 += 1
        stnpresmaxtotal += stnpresmax
    if (df.iloc[k,2] != -999.0):
        stnpresmin = float(df.iloc[k,2])
        count2 += 1
        stnpresmintotal += stnpresmin
    if (df.iloc[k,3] != -999.0):
        T = float(df.iloc[k,3])
        count3 += 1
        Ttotal += T
    if (df.iloc[k,4] != -999.0):
        Tmax = float(df.iloc[k,4])
        count4 += 1
        Tmaxtotal += Tmax
    if (df.iloc[k,5] != -999.0):
        Tmin = float(df.iloc[k,5])
        count5 += 1
        Tmintotal += Tmin
    if (df.iloc[k,8] != -999.0):
        WS = float(df.iloc[k,8])
        count8 += 1
        WStotal += WS
    if (df.iloc[k,10] != -999.0):
        WSGust = float(df.iloc[k,10])
        count10 += 1
        WSGusttotal += WSGust

ave0 = round(stnprestotal / count0 , 1)
ave1 = round(stnpresmaxtotal / count1 , 1)
ave2 = round(stnpresmintotal / count2 , 1)
ave3 = round(Ttotal / count3 , 1)
ave4 = round(Tmaxtotal / count4 , 1)
ave5 = round(Tmintotal / count5 , 1)
ave8 = round(WStotal / count8 , 1)
ave10 = round(WSGusttotal / count10 , 1)

for c in range(854):
    if df.iloc[c,0] == -999.0:
        df.iloc[c,0] = ave0
    if df.iloc[c,1] == -999.0:
        df.iloc[c,1] = ave1
    if df.iloc[c,2] == -999.0:
        df.iloc[c,2] = ave2
    if df.iloc[c,3] == -999.0:
        df.iloc[c,3] = ave3
    if df.iloc[c,4] == -999.0:
        df.iloc[c,4] = ave4
    if df.iloc[c,5] == -999.0:
        df.iloc[c,5] = ave5
    if df.iloc[c,8] == -999.0:
        df.iloc[c,8] = ave8
    if df.iloc[c,10] == -999.0:
        df.iloc[c,10] = ave10

for i in range(854):
    if df.iloc[i,6] == -999.0:
            df.iloc[i,6] = df['RH'].value_counts().idxmax()

for i in range(854):
    if df.iloc[i,7] == -999.0:
            df.iloc[i,7] = df['RHMin'].value_counts().idxmax()           

for i in range(854):
    if df.iloc[i,9] == -999.0:
            df.iloc[i,9] = df['WD'].value_counts().idxmax()

for i in range(854):
    if df.iloc[i,11] == -999.0:
            df.iloc[i,11] = df['WDGust'].value_counts().idxmax()

X = df.drop(['Precp'], axis=1)
y = df['Precp']

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

X_train,X_test, y_train,y_test = train_test_split(X, y, test_size=0.3, random_state=67)
lr = LogisticRegression(max_iter=200)
lr.fit(X_train, y_train)
predictions = lr.predict(X_test)
print(accuracy_score(y_test, predictions))
print(recall_score(y_test, predictions))
print(precision_score(y_test, predictions))

X = df.drop(['Precp'], axis=1)
y = df['Precp']
X_train,X_test, y_train,y_test = train_test_split(X, y, test_size=0.3, random_state=67)
lr = LogisticRegression(max_iter=200)
lr.fit(X_train, y_train)
predictions = lr.predict(X_test)
print(accuracy_score(y_test, predictions))
print(recall_score(y_test, predictions))
print(precision_score(y_test, predictions))

pd.DataFrame(confusion_matrix(y_test, predictions), columns = ['Predict not rain','Predict rain'], index=['True not rain', 'True rain'])

print(lr.predict([[900, 1000, 850, 23, 27, 18, 34, 12, 1, 23, 2, 45]]))
print(lr.predict([[900, 860, 950 , 26, 31, 20, 70 , 50 , 3 , 20 , 6 , 25 ]]))

import joblib
joblib.dump(lr,'Precipitation_Predict_1.pkl',compress=3)
print('Finish')