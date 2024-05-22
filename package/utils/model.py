import subprocess
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
from package.utils.utils import set_or_create_experiment, get_performance_plots_regr

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

def predict(model_uri, input_data):
    model = mlflow.pyfunc.load_model(model_uri)
    predictions = model.predict(input_data)
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
    model = mlflow.pyfunc.load_model(model_uri)

    y_true = new_data[target]
    X_new = new_data.drop(columns=[target])
    y_pred = model.predict(X_new)

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
