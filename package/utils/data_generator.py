import time
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import os


def generate_random_data_with_trends(num_rows, data, out_data):
    df = pd.read_csv(data, sep=",")
    ranges = {
        'RVI': (df['RVI'].min(), df['RVI'].max()),
        'dust': (df['dust'].min(), df['dust'].max()),
        'humidity': (df['humidity'].min(), df['humidity'].max()),
        'light': (df['light'].min(), df['light'].max()),
        'noise': (df['noise'].min(), df['noise'].max()),
        'pressure': (df['pressure'].min(), df['pressure'].max()),
        'radon': (df['radon'].min(), df['radon'].max()),
        'score': (df['score'].min(), df['score'].max()),
        'temp': (df['temp'].min(), df['temp'].max()),
        'voc': (df['voc'].min(), df['voc'].max()),
        'co2': (df['co2'].min(), df['co2'].max())
    }
    start_date = datetime(2024, 5, 1, 0, 0, 0)
    time_interval_seconds = 1

    def sequential_date(start, period, index):
        return start + timedelta(hours=period * index)

    for i in range(num_rows):
        trend_factor = i / num_rows
        row = {
            'RVI': np.random.randint(*ranges['RVI']),
            'dust': np.random.uniform(*ranges['dust']) + trend_factor * 2,
            'humidity': np.random.uniform(*ranges['humidity']) + trend_factor * 10,
            'light': np.random.uniform(*ranges['light']),
            'noise': np.random.uniform(*ranges['noise']) + trend_factor * 10,
            'pressure': np.random.uniform(*ranges['pressure']) + trend_factor * 5,
            'radon': np.random.randint(*ranges['radon']),
            'score': np.random.randint(*ranges['score']),
            'temp': np.random.uniform(*ranges['temp']) + trend_factor * 2,
            'voc': np.random.randint(*ranges['voc']) + trend_factor * 10,
            'co2': np.random.randint(*ranges['co2'])
        }
        row_df = pd.DataFrame([row])
        row_df.to_csv(out_data, mode='a', sep=',', index=False, header=not os.path.exists(out_data))
        time.sleep(time_interval_seconds)