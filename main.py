# -*- coding: utf-8 -*-
'''
Create Date: 2023/08/24
Author: @1chooo(Hugo ChunHo Lin)
Version: v0.0.3
'''

from os.path import join
from os.path import dirname
from os.path import abspath
import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from Plum.Utils.Tools import load_data
from Plum.Utils.Tools import save_model
from Plum.Utils.Tools import test_model
from Plum.Utils.Plot import plot_confusion_matrix
from Plum.Utils.Plot import plot_evaluation_metrics
from Plum.Utils.Plot import plot_predicted_probability_distribution
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

df = load_data()

# Drop irrelevant columns
columns_to_drop = [
    'ObsTime'       , 'SeaPres'       , 'StnPresMaxTime', 
    'StnPresMinTime', 'T Max Time'    , 'T Min Time'    , 
    'Td dew point'  , 'RHMinTime'     , 'WGustTime'     , 
    'PrecpHour'     , 'PrecpMax10'    , 'PrecpMax10Time', 
    'PrecpMax60'    , 'PrecpMax60Time', 'SunShine'      , 
    'SunShineRate'  , 'GloblRad'      , 'VisbMean'      , 
    'EvapA'         , 'UVI Max'       , 'UVI Max Time'  ,  
    'Cloud Amount'
]
df.drop(columns_to_drop, axis=1, inplace=True)

# Replace missing values
df.replace(['...', '/'], '-999', inplace=True)
df.replace('-999', 0.0, inplace=True)

# Convert DataFrame to float64
df = pd.DataFrame(df, dtype = np.float64)

for k in range(len(df)):
    if df.iloc[k, 12] > 0.0:
        df.iloc[k, 12] = 1
    else:
        df.iloc[k, 12] = 0

# Initialize counters and totals
observation_counts = [0, 0, 0, 0, 0, 0, 0, 0]
observation_totals = [0, 0, 0, 0, 0, 0, 0, 0]

# Define the columns and their corresponding indices
observed_columns_indices = [0, 1, 2, 3, 4, 5, 8, 10]

# Iterate over each row in the DataFrame
for row_index in range(len(df)):
    for col_index, obs_col_idx in enumerate(observed_columns_indices):
        if df.iloc[row_index, obs_col_idx] != -999.0:
            value = float(df.iloc[row_index, obs_col_idx])
            observation_counts[col_index] += 1
            observation_totals[col_index] += value

# Calculate averages
observed_averages = [
    round(total / count, 1) if count > 0 else 0 for total, count in zip(observation_totals, observation_counts)
]

# Assign averaged values to descriptive variables
stn_avg, stnmax_avg, stnmin_avg, WS_avg, WSGust_avg, T_avg, Tmax_avg, Tmin_avg = observed_averages

# Define a list of column indices that need to be filled with averages
columns_to_fill = [0, 1, 2, 3, 4, 5, 8, 10]

# Define a list of corresponding average values
average_values = [
    stn_avg   , stnmax_avg,
    stnmin_avg, WS_avg    ,
    WSGust_avg, T_avg     ,
    Tmax_avg  , Tmin_avg  ,
]

# Iterate over the rows in the DataFrame
for c in range(len(df)):
    for col_idx, avg_value in zip(columns_to_fill, average_values):
        if df.iloc[c, col_idx] == -999.0:
            df.iloc[c, col_idx] = avg_value

columns_to_fill = [
    (6, 'RH'), 
    (7, 'RHMin'), 
    (9, 'WD'), 
    (11, 'WDGust')
]

for col_idx, col_name in columns_to_fill:
    for i in range(len(df)):
        if df.iloc[i, col_idx] == -999.0:
            df.iloc[i, col_idx] = df[col_name].value_counts().idxmax()

def train_logistic_regression(df):
    X = df.drop(['Precp'], axis=1)
    y = df['Precp']

    X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y, 
        test_size=0.3, 
        random_state=67,
    )
    
    lr = LogisticRegression(max_iter=200)
    lr.fit(X_train, y_train)

    predictions = lr.predict(X_test)

    return lr, X_test, y_test, predictions

def evaluate_model(y_test, predictions):
    accuracy = accuracy_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    confusion = pd.DataFrame(
        confusion_matrix(y_test, predictions), 
        columns=['Predict not rain', 'Predict rain'], 
        index=['True not rain', 'True rain']
    )

    return accuracy, recall, precision, confusion

lr, X_test, y_test, predictions = train_logistic_regression(df)
accuracy, recall, precision, confusion = evaluate_model(y_test, predictions)

print("Accuracy:", accuracy)
print("Recall:", recall)
print("Precision:", precision)
print("Confusion Matrix:\n", confusion)

proba = lr.predict_proba(X_test)[:, 1]

plot_confusion_matrix(confusion)
plot_evaluation_metrics(accuracy, recall, precision)
plot_predicted_probability_distribution(proba)

first_test = test_model(lr, 900, 1000, 850, 23, 27, 18, 34, 12, 1, 23, 2, 45)
second_test = test_model(lr, 900, 860, 950, 26, 31, 20, 70, 50, 3, 20, 6, 25)

print('First:', first_test)
print('Second:', second_test)

output_model_path = join(
    dirname(abspath(__file__)),
    'model', 
    'plum_prediction.pkl'
)

save_model(lr, output_model_path, 3)
