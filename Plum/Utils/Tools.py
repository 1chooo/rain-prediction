# -*- coding: utf-8 -*-
'''
Create Date: 2023/08/24
Author: @1chooo(Hugo ChunHo Lin)
Version: v0.0.3
'''

import os
import pandas as pd
from glob import glob
from os.path import join
from os.path import dirname
from os.path import abspath
import joblib

def load_data() -> pd.DataFrame:
    all_data_path = join(
        dirname(abspath(__file__)),
        '..', 
        '..', 
        'data',
        '*.csv',
    )

    csv_files = glob(all_data_path)
    csv_files.sort()

    dfs_to_merge = []  # List to store DataFrames for merging
    pivot = 2016

    for file in csv_files:
        file_name = os.path.basename(file)
        year = int(file_name.split('-')[1])

        delimiter = '\t' if year >= pivot else ','  # Choose delimiter based on the year

        df = pd.read_csv(file, delimiter=delimiter)
        dfs_to_merge.append(df)

    merged_df = pd.concat(dfs_to_merge, ignore_index=True)  # Concatenate DataFrames

    return merged_df

def output_merged_data() -> None:
    merged_df = load_data()

    output_path = join(
        dirname(abspath(__file__)),
        '..', 
        '..', 
        'data',
        'merged_data',
        '20_years_data.csv',
    )
    merged_df.to_csv(
        output_path, 
        sep=',', 
        index=False, 
        encoding='utf-8-sig'
    )

    print("Merging and transforming multiple CSV files completed.")

def save_model(lr, filename, compress):
    # Get the directory path for the model file
    model_directory = dirname(abspath(filename))
    # Check if the directory exists, create if not
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)

    joblib.dump(lr, filename, compress=compress)
