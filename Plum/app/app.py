# -*- coding: utf-8 -*-
'''
Create Date: 2023/08/26
Author: @1chooo(Hugo ChunHo Lin)
Version: v0.0.4
'''

from os.path import join
from os.path import dirname
from os.path import abspath
import gradio as gr
from Plum.app.Train import train
from Plum.Utils.Tools import call_model

def start_gradio():
    train()
    model_path = join(
        dirname(abspath(__file__)),
        '..',
        '..',
        'model', 
        'plum_prediction.pkl'
    )
    call_model(model_path, 900, 1000, 850, 23, 27, 18, 34, 12, 1, 23, 2, 45)
    call_model(model_path, 900, 860 , 950, 26, 31, 20, 70, 50, 3, 20, 6, 25)