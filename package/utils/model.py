import subprocess
import time
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import mlflow
import mlflow.sklearn
from package.utils.utils import set_or_create_experiment, get_performance_plots_regr

def train_and_log_model(X, y, experiment_name="experiment"):
    experiment_id = set_or_create_experiment(experiment_name)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create the LightGBM dataset
    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test)

    # Define parameters
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9
    }

    model = lgb.train(params, train_data, valid_sets=[test_data])
    y_pred = model.predict(X_test, num_iteration=model.best_iteration)


    performance_results = get_performance_plots_regr(y_test, y_pred)
    mse = performance_results["mse"]
    with mlflow.start_run(run_name="run") as run:
        mlflow.sklearn.log_model(model, "model")
        mlflow.log_params(params)
        mlflow.log_metric("MSE", mse)
        mlflow.log_metric("MAE", performance_results["mae"])
        mlflow.log_metric("R2", performance_results["r2"])
        mlflow.log_figure(performance_results["true_vs_pred"], "true_vs_pred.png")
        mlflow.log_figure(performance_results["residual_plot"], "residual_plot.png")

        return mlflow.active_run().info.run_id, mse

def predict(model_uri, input_data):
    model = mlflow.pyfunc.load_model(model_uri)
    predictions = model.predict(input_data, num_iteration=model.best_iteration)
    return predictions


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


def monitor_and_retrain(new_data, current_run_id, active_process, target, threshold=0.05):
    model_uri = f"runs:/{current_run_id}/model"
    y_true = new_data[target]
    X_new = new_data.drop(columns=[target])
    y_pred = predict(model_uri, X_new)

    new_accuracy = mean_squared_error(y_true, y_pred)
    print(f"New MSE: {new_accuracy}")

    old_accuracy = mlflow.get_run(current_run_id).data.metrics['MSE']
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
