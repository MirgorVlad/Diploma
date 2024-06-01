import os

from flask import Flask, request, jsonify
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from package.utils.model import train_and_log_model, initial_training

app = Flask(__name__)

# Path to save the model
MODEL_PATH = "/mnt/models/model.pkl"


# Load the model
def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
    else:
        model = initial_training()

load_model()


@app.route('/predict', methods=['POST'])
def predict():
    global model
    data = request.get_json(force=True)
    df = pd.DataFrame(data['data']['ndarray'])
    predictions = model.predict(df)
    return jsonify(predictions.tolist())


@app.route('/retrain', methods=['POST'])
def retrain():
    global model
    file = request.files['file']
    data = pd.read_csv(file)

    # Assuming the last column is the target
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    run_id, mse, model = train_and_log_model(X, y)

    joblib.dump(model, MODEL_PATH)
    load_model()

    return jsonify({"status": "Model retrained", "mse": mse})


@app.route('/statistics', methods=['GET'])
def statistics():
    experiment = mlflow.get_experiment_by_name("experiment")
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    stats = runs[["metrics.mse", "start_time"]].to_dict(orient="records")
    return jsonify(stats)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
