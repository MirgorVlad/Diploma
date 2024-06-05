import pandas as pd
import numpy as np
import requests
import time
import csv
import os

from package.utils.model import check_degradation

MODEL_URL = "http://127.0.0.1:5000/"
DATA_CSV_PATH = "data/sensor_data.csv"
OUT_CSV_FILE_PATH = "data/out_data.csv"

df = pd.read_csv(DATA_CSV_PATH, sep=",")

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

TARGET_THRESHOLD_MAX = 900
TARGET_THRESHOLD_MIN = 400
def generate_random_data_with_trends(num_rows):
    for i in range(num_rows):
        trend_factor = i / num_rows
        row_data = {
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
        data = pd.DataFrame([row_data])
        if not os.path.exists(OUT_CSV_FILE_PATH):
            data.to_csv(OUT_CSV_FILE_PATH, mode='w', sep=',', index=False)
        input = data.drop(columns=["co2"])
        for index, row in input.iterrows():

            payload = {
                'features': ','.join(input.columns),
                'values': ','.join(map(str, row.values))
            }
            print("data: ", payload)

            # Send data to the prediction endpoint
            prediction = send_prediction_request(payload)
            print("prediction: ", prediction, "\n")

            if(prediction > TARGET_THRESHOLD_MAX or prediction < TARGET_THRESHOLD_MIN):
                print("Model reach prediction threshold!")
                statistics = send_statistics_request()
                current_mse = statistics["MSE"]
                retrain_data = send_retrain_request(OUT_CSV_FILE_PATH)
                new_mse = retrain_data["MSE"]
                if(check_degradation(current_mse, new_mse)):
                    send_switch_request()

            data.to_csv(OUT_CSV_FILE_PATH, mode='a', sep=',', index=False, header=False)
            # Wait before sending the next request
        time.sleep(1)  # wait 1 second before sending the next data point


def send_switch_request(payload):
    response = requests.post(MODEL_URL + "switch", json=payload)
    print(response.json())

def send_prediction_request(payload):
    response = requests.post(MODEL_URL + "predict", json=payload)
    prediction = response.json()[0]
    return prediction

def send_retrain_request(file_path):
    with open(file_path, 'rb') as file:
        response = requests.post(MODEL_URL + "retrain", files={'file': file})
        if response.status_code == 200:
            print("Model retrained successfully")
            return response.json()
        else:
            return {"status": "Error", "message": response.text}

def send_statistics_request():
    response = requests.get(MODEL_URL + "statistics")
    statistics = response.json()
    return statistics

if __name__ == "__main__":
    num_rows = 100
    generate_random_data_with_trends(num_rows)

#     if not os.path.exists(OUT_CSV_FILE_PATH):
#        input.to_csv(OUT_CSV_FILE_PATH, mode='w', sep=',', index=False)

# row.to_frame().T.to_csv(OUT_CSV_FILE_PATH, mode='a', sep=',', index=False, header=False)