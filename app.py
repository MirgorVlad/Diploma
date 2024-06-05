import os
from flask import Flask, request, jsonify
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from package.utils.model import train_and_log_model, initial_training, get_runs

# Path to save the model
MODEL_PATH = "./models/model.pkl"

# Global model variable
model = None

# Load the model
def load_model():
    global model
    if model is None:
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
        else:
            model = initial_training()


app = Flask(__name__)
load_model()

@app.route('/predict', methods=['POST'])
def predict():
    global model
    data = request.get_json(force=True)

    # Extract feature names and values
    features = data['features'].split(',')
    values = [float(x) for x in data['values'].split(',')]

    # Convert to DataFrame
    df = pd.DataFrame([values], columns=features)

    # Make predictions
    predictions = model.predict(df, num_iteration=model.best_iteration)
    return jsonify(predictions.tolist())

@app.route('/retrain', methods=['POST'])
def retrain():
    global model
    file = request.files['file']
    df = pd.read_csv(file)

    target = "co2"
    X = df.drop(columns=[target])
    y = df[target]

    run_id, mse, model = train_and_log_model(X, y)

    joblib.dump(model, MODEL_PATH)
    load_model()

    return jsonify({"status": "Model retrained", "mse": mse})

@app.route('/statistics', methods=['GET'])
def statistics():
    runs = get_runs()
    latest_run = runs.iloc[0]
    stats = {
        "run_id": latest_run.run_id,
        "MSE": latest_run["metrics.MSE"],
        "MAE": latest_run["metrics.MAE"],
        "R2": latest_run["metrics.R2"],
        "start_time": latest_run.start_time
    }
    return jsonify(stats)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001)
