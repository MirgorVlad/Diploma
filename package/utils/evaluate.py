import datetime

import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from package.feature.data_processing import preprocess_data
from package.ml_training.retrieal import get_train_test_set
from package.ml_training.train import train_model
from package.utils.utils import get_performance_plots_regr


def load_model(model_name):
    model_uri = f"models:/{model_name}/latest"
    return mlflow.sklearn.load_model(model_uri=model_uri)


def preprocess_device_data(row):
    df = preprocess_data(pd.DataFrame(row))
    return df.drop(columns=['Timestamp', 'noise', 'dust'])

def check_and_retrain(threshold, prediction):
    if prediction < threshold:
        retrain_model()

def retrain_model():
    model_name = "rfm"
    experiment_name = "co2 prediction"
    run_name = "run01"
    artifact_path = "artifacts"
    prefix = "retrain"
    data = pd.read_csv("data/generated_sensor_data.csv", sep=";")
    df = preprocess_data(data)

    X_train, X_test, y_train, y_test = get_train_test_set(df)
    n_estimators = 100
    random_state = 42

    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.register_model(model_uri=f"runs:/{run.info.run_id}/{artifact_path}", name=model_name)
        model.fit(X_train, y_train)
        # mlflow.sklearn.log_model(model, model_name)
        y_pred = model.predict(X_test)

        performance_results = get_performance_plots_regr(y_test, y_pred, "test")


        # Log performance metrics

        # Log parameters
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("random_state", random_state)

        # Set tags
        mlflow.set_tag("experiment_name", experiment_name)
        mlflow.set_tag("run_name", run_name)

        # Log performance metrics
        mlflow.log_metric("MSE", performance_results["mse"])
        mlflow.log_metric("MAE", performance_results["mae"])
        mlflow.log_metric("R2", performance_results["r2"])

        # Log figures with MLflow
        mlflow.log_figure(performance_results[f"{prefix}_true_vs_pred"], prefix + "_true_vs_pred.png")
        mlflow.log_figure(performance_results[f"{prefix}_residual_plot"], prefix + "_residual_plot.png")
        print(f"Model retrained and logged in run {run.info.run_id}")

def log_prediction(row, predicted):
    with mlflow.start_run(run_name="batch_logging") as run:
            # Log each row's metrics
            mlflow.log_metric("RVI", row['RVI'])
            mlflow.log_metric("humidity", row['humidity'])
            mlflow.log_metric("light", row['light'])
            mlflow.log_metric("pressure", row['pressure'])
            mlflow.log_metric("radon", row['radon'])
            mlflow.log_metric("temp", row['temp'])
            mlflow.log_metric("voc", row['voc'])
            mlflow.log_metric("co2", predicted)