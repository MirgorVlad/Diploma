import os
import time

import numpy as np
from datetime import datetime, timedelta
from package.utils.evaluate import *


SENSOR_DATA = "data/sensor_data.csv"
OUT_DATA = "data/generated_sensor_data.csv"

# Read the data into a pandas DataFrame
df = pd.read_csv(SENSOR_DATA, sep=";")


# Define a function to generate random dates in sequence
def sequential_date(start, period, index):
    return (start + timedelta(hours=period * index)).strftime("%Y-%m-%d %H:%M:%S")


# Get the range of each column
ranges = {
    'RVI': (df['RVI'].min(), df['RVI'].max()),
    'co2': (df['co2'].min(), df['co2'].max()),
    'dust': (df['dust'].min(), df['dust'].max()),
    'humidity': (df['humidity'].min(), df['humidity'].max()),
    'light': (df['light'].min(), df['light'].max()),
    'noise': (df['noise'].min(), df['noise'].max()),
    'pressure': (df['pressure'].min(), df['pressure'].max()),
    'radon': (df['radon'].min(), df['radon'].max()),
    'score': (df['score'].min(), df['score'].max()),
    'temp': (df['temp'].min(), df['temp'].max()),
    'voc': (df['voc'].min(), df['voc'].max())
}

# Define the start date and time interval (e.g., 1 hour)
start_date = datetime.datetime(2024, 5, 1, 0, 0, 0)
time_interval_seconds = 1


# Function to generate random data with trends
def generate_random_data_with_trends(num_rows, model):
    for i in range(num_rows):
        # Simulate some telemetry trend over time
        trend_factor = i / num_rows

        row = {
            'Timestamp': sequential_date(start_date, time_interval_seconds, i),
            'RVI': np.random.randint(*ranges['RVI']),
            'dust': np.random.uniform(*ranges['dust']) + trend_factor * 2,  # Increasing trend for dust
            'humidity': np.random.uniform(*ranges['humidity']) + trend_factor * 10,
            'light': np.random.uniform(*ranges['light']),
            'noise': np.random.uniform(*ranges['noise']) + trend_factor * 10,  # Increasing trend for noise
            'pressure': np.random.uniform(*ranges['pressure']) + trend_factor * 5,
            'radon': np.random.randint(*ranges['radon']),
            'score': np.random.randint(*ranges['score']),
            'temp': np.random.uniform(*ranges['temp']) + trend_factor * 2,  # Increasing trend for temperature
            'voc': np.random.randint(*ranges['voc']) + trend_factor * 10  # Increasing trend for VOC
        }
        df = preprocess_device_data([row])
        prediction = model.predict(df)
        row["co2"] = prediction[0]
        pd.Series(row).to_frame().T.to_csv(OUT_DATA, mode='a', sep=';', index=False,
                                           header=not os.path.exists(OUT_DATA))

        log_prediction(row, prediction[0])
        print(row, "\n",  prediction[0])
        check_and_retrain(prediction=prediction, threshold=500)
        time.sleep(time_interval_seconds)

if __name__ == '__main__':
    model = load_model("rfm")
    generate_random_data_with_trends(100, model)
