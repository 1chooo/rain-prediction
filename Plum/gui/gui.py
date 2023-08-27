# -*- coding: utf-8 -*-
'''
Create Date: 2023/08/24
Author: @1chooo(Hugo ChunHo Lin)
Version: v0.0.5
'''

import gradio as gr
from typing import Any
import textwrap
from Plum.Utils.Tools import get_predict_result

def build_ui():
    demo = gr.Blocks(
        title='梅雨還是沒雨？',
    )

    with demo:
        with gr.Row():
            our_heading = heading()
        with gr.Tab("歡迎輸入數據"):
            playground = generate_playground()
        with gr.Tab("模型訓練過程"):
            gr.Markdown(f"模型訓練過程")

    demo.launch(
        # enable_queue=True,
        # share=True, 
        server_name="127.0.0.1", 
        server_port=6006,
        debug=True,
    ) 

def heading(*args: Any, **kwargs: Any) -> gr.Markdown:
    title = '梅雨還是沒雨？'
    descriptions = textwrap.dedent(
    """
    我們是一群大氣系的學生，我們想運用大氣的知識結合機器學習的技術來預測降雨的可能性，\
    即便如今大氣預測降雨的機率已行之有年，不過我們依舊願意嘗試新的方法，\
    就如同其他機器學習的預測像是我們嘗試過的鐵達尼號生存預測、房價預測、借貸評估一般，\
    在已知的數據中，得出更接近真實情況的結果預測。
    """
    )

    our_heading = gr.Markdown(
        f"""\
        # {title}
        {descriptions}
        """
    )

    return our_heading

def generate_playground():
    with gr.Row() as playground:
        gr.Markdown(
            textwrap.dedent(
            """
            以下有些天氣數值可以提供給大家使用，歡迎輸入各種數據！！！
            """
            )
        )
        
    with gr.Row() as playground:
        gr.Markdown(f"## 請輸入氣壓資訊")

    with gr.Row():
        current_pressure = gr.Slider(
            800, 1050, value=1000, label="當前氣壓值", info="800.0~1050"
        )
        today_max_pressure = gr.Slider(
            800, 1050, value=1000, label="本日最低氣壓值", info="800.0~1050"
        )
        today_min_pressure = gr.Slider(
            800, 1050, value=1000, label="本日最高氣壓值", info="800.0~1050"
        )

    with gr.Row():
        gr.Markdown(f"## 請輸入溫度資訊")
    with gr.Row():
        current_temperature = gr.Slider(
            0, 40, value=20, label="當前溫度值", info="0~40"
        )
        today_max_temperature = gr.Slider(
            0, 40, value=20, label="本日最高溫度值", info="0~40"
        )
        today_min_temperature = gr.Slider(
            0, 40, value=20, label="本日最低溫度值", info="0~40"
        )

    with gr.Row():
        gr.Markdown(f"## 輸入相對濕度資訊")
    with gr.Row():
        current_relative_humidity = gr.Slider(
            0, 100, value=20, label="當前相對濕度", info="0~100"
        )
        today_min_relative_humidity = gr.Slider(
            0, 100, value=20, label="本日最低相對濕度", info="0~100"
        )

    with gr.Row():
        gr.Markdown(f"## 請輸入風資訊")
    with gr.Row():
        current_wind_speed = gr.Slider(
            0, 10, value=5, label="當前風速", info="0.0~10.0"
        )
        current_wind_direction = gr.Slider(
            1, 360, value=20, label="當前風向", info="1~360"
        )
    with gr.Row():
        gr.Markdown("## 請輸入陣風資訊")
    with gr.Row():
        current_gust_wind_speed = gr.Slider(
            0, 10, value=5, label="當前陣風風速", info="0.0~10.0"
        )
        current_gust_wind_direction = gr.Slider(
            1, 360, value=20, label="當前陣風風向", info="1~360"
        )
    with gr.Row():
        predict_result = gr.Text(
            label="我們評估的結果",
            type="text",
        )
        predict_confidence = gr.Text(
            label="我們評估的系統信心程度",
            type="text",
        )
    with gr.Row():
        submit_set_btn = gr.Button(value="Submit Setting")
        submit_set_btn.click(
            fn=get_predict_result, 
            inputs=[
                current_pressure, today_max_pressure, today_min_pressure, 
                current_temperature, today_max_temperature, today_min_temperature, 
                current_relative_humidity, today_min_relative_humidity,
                current_wind_speed, current_wind_direction,
                current_gust_wind_speed, current_gust_wind_direction
            ], 
            outputs=[predict_result, predict_confidence]
        )
    gr.Examples(
        [
            [900, 1000, 850, 23, 27, 18, 34, 12, 1, 23, 2, 45],
            [900, 860 , 950, 26, 31, 20, 70, 50, 3, 20, 6, 25],
        ],
        inputs=[
                current_pressure, today_max_pressure, today_min_pressure, 
                current_temperature, today_max_temperature, today_min_temperature, 
                current_relative_humidity, today_min_relative_humidity,
                current_wind_speed, current_wind_direction,
                current_gust_wind_speed, current_gust_wind_direction
        ], 
        outputs=[predict_result, predict_confidence],
        fn=get_predict_result,
        # cache_examples=True,
        )
    return playground