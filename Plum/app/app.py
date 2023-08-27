# -*- coding: utf-8 -*-
'''
Create Date: 2023/08/26
Author: @1chooo(Hugo ChunHo Lin)
Version: v0.0.5
'''

import gradio as gr
from Plum.Model.Train import train
from Plum.gui.gui import build_ui

def start_gradio():
    train()
    build_ui()
