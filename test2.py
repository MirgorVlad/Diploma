import subprocess
import time
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
import os
import requests

from package.feature.data_processing import load_data
from package.utils.utils import set_or_create_experiment, get_performance_plots_regr

INITIAL_PORT = 5001
SENSOR_DATA = "data/sensor_data.csv"
OUT_DATA = "data/new_data.csv"


def train_and_log_model(X, y, experiment_name="experiment"):
    experiment_id = set_or_create_experiment(experiment_name)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=20, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    performance_results = get_performance_plots_regr(y_test, y_pred)

    with mlflow.start_run(run_name="run") as run:
        mlflow.sklearn.log_model(sk_model=model, artifact_path="model")
        mlflow.log_param("n_estimators", model.n_estimators)
        mlflow.log_param("max_depth", model.max_depth)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("MSE", performance_results["mse"])
        mlflow.log_metric("MAE", performance_results["mae"])
        mlflow.log_metric("R2", performance_results["r2"])
        mlflow.log_figure(performance_results["true_vs_pred"], "true_vs_pred.png")
        mlflow.log_figure(performance_results["residual_plot"], "residual_plot.png")

        return mlflow.active_run().info.run_id, accuracy


def generate_random_data_with_trends(num_rows):
    df = pd.read_csv(SENSOR_DATA, sep=";")
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

    generated_data = []
    for i in range(num_rows):
        trend_factor = i / num_rows
        row = {
            'Timestamp': sequential_date(start_date, time_interval_seconds, i),
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
            'co2': np.random.randint(*ranges['co2']) + trend_factor * 10
        }
        generated_data.append(row)
        time.sleep(time_interval_seconds)

    new_data_df = pd.DataFrame(generated_data)
    new_data_df.to_csv(OUT_DATA, mode='a', sep=';', index=False, header=not os.path.exists(OUT_DATA))


def predict_rest_api(port, input_data):
    url = f"http://127.0.0.1:{port}/invocations"
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, json=input_data, headers=headers)
    return response.json()


def deploy_model(run_id, port):
    model_uri = f"runs:/{run_id}/model"
    cmd = f"mlflow models serve --model-uri {model_uri} --port {port} --no-conda"
    process = subprocess.Popen(cmd, shell=True)
    return process


def stop_model(process):
    process.terminate()


def get_active_model_port():
    try:
        with open("active_model.txt", "r") as f:
            port = f.read().strip()
        return int(port)
    except FileNotFoundError:
        return None


def set_active_model_port(port):
    with open("active_model.txt", "w") as f:
        f.write(str(port))


def monitor_and_retrain(new_data, current_run_id, active_process, threshold=0.05):
    active_port = get_active_model_port()
    X_new = new_data.drop(columns=['co2', 'Timestamp'])
    y_true = new_data['co2'].tolist()  # Convert to list for JSON serialization

    # Prepare input data for the REST API
    input_data = {
        "columns": X_new.columns.tolist(),
        "data": X_new.values.tolist()
    }

    # Get predictions from the REST API
    y_pred = predict_rest_api(active_port, input_data)
    new_accuracy = accuracy_score(y_true, y_pred)
    print(f"New Accuracy: {new_accuracy}")

    old_accuracy = mlflow.get_run(current_run_id).data.metrics['accuracy']
    if (old_accuracy - new_accuracy) > threshold:
        print("Retraining model...")
        new_run_id, new_accuracy = train_and_log_model(X_new, y_true)
        print(f"New model accuracy: {new_accuracy}")

        if new_accuracy > old_accuracy:
            print(f"Switching to new model with run id {new_run_id}")
            active_port = get_active_model_port()
            if active_port == 5001:
                new_port = 5002
            else:
                new_port = 5001

            new_process = deploy_model(new_run_id, new_port)
            time.sleep(10)  # Give it some time to start up

            if active_port:
                stop_model(active_process)

            set_active_model_port(new_port)
            return new_run_id, new_process
        else:
            print("New model did not improve, keeping the current model.")
    else:
        print("Current model is still accurate.")

    return current_run_id, active_process


if __name__ == "__main__":
    # Initial training and logging
    df = load_data()
    target = "co2"
    X = df.drop(columns=[target, "Timestamp"], axis=1)
    y = df[target]
    initial_run_id, initial_accuracy = train_and_log_model(X, y)
    print(f"Initial model accuracy: {initial_accuracy}")

    # Start data generation and monitoring loop
    active_process = deploy_model(initial_run_id, INITIAL_PORT)
    set_active_model_port(INITIAL_PORT)

    while True:
        generate_random_data_with_trends(100)
        new_data = pd.read_csv(OUT_DATA, sep=";")
        current_run_id, active_process = monitor_and_retrain(new_data, initial_run_id, active_process)
        initial_run_id = current_run_id  # Update to the latest model run id
        time.sleep(60)  # Monitor every minute
