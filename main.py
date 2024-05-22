import time

import pandas as pd

from package.feature.data_processing import load_data
from package.utils.data_generator import generate_random_data_with_trends
from package.utils.model import train_and_log_model, deploy_model, set_active_model_port, monitor_and_retrain

SENSOR_DATA = "data/sensor_data.csv"
OUT_DATA = "data/new_data.csv"
GREEN_PORT = 5001

if __name__ == "__main__":
    # Initial training and logging
    df = load_data()
    target = "co2"

    X = df.drop(columns=[target])
    y = df[target]
    initial_run_id, initial_accuracy = train_and_log_model(X, y)
    print(f"Initial model accuracy: {initial_accuracy}")

    # Start data generation and monitoring loop
    active_process = deploy_model(initial_run_id, GREEN_PORT)
    set_active_model_port(GREEN_PORT)

    while True:
        generate_random_data_with_trends(100, SENSOR_DATA, OUT_DATA)
        new_data = pd.read_csv(OUT_DATA, sep=",")
        current_run_id, active_process = monitor_and_retrain(new_data, initial_run_id, active_process, 0.05)
        initial_run_id = current_run_id  # Update to the latest model run id
        time.sleep(60)  # Monitor every minute