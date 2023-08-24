# -*- coding: utf-8 -*-
'''
Create Date: 2023/08/24
Author: @1chooo(Hugo ChunHo Lin)
Version: v0.0.2
'''

import os
import pandas as pd
from glob import glob
from os.path import join
from os.path import dirname
from os.path import abspath

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

    merged_df = pd.DataFrame()

    for file in csv_files:
        file_name = os.path.basename(file)
        year = int(file_name.split('-')[1])

        if year >= 2016:
            df = pd.read_csv(file, delimiter='\t')
        else:
            df = pd.read_csv(file, delimiter=',')

        merged_df = merged_df.append(df, ignore_index=True)
    
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
