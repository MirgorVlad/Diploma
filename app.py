import os
from flask import Flask, request, jsonify
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from package.utils.model import train_and_log_model, initial_training, get_run

# Path to save the models
MODEL_DIR = "./models"
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
LAST_RUN_ID = ""
CURRENT_RUN_ID = ""
# Global model variables
current_model = None
new_model = None

# Load models
def load_models():
    global current_model, new_model, CURRENT_RUN_ID
    if current_model is None:
        current_model_path = os.path.join(MODEL_DIR, "current_model.pkl")
        if os.path.exists(current_model_path):
            current_model = joblib.load(current_model_path)
        else:
            current_model, run_id = initial_training()
            CURRENT_RUN_ID = run_id
            joblib.dump(current_model, current_model_path)

    if new_model is None:
        new_model_path = os.path.join(MODEL_DIR, "new_model.pkl")
        if os.path.exists(new_model_path):
            new_model = joblib.load(new_model_path)
        else:
            new_model = None

app = Flask(__name__)
load_models()

@app.route('/predict', methods=['POST'])
def predict():
    global current_model, new_model
    data = request.get_json(force=True)

    # Extract feature names and values
    features = data['features'].split(',')
    values = [float(x) for x in data['values'].split(',')]

    # Convert to DataFrame
    df = pd.DataFrame([values], columns=features)

    # Choose model based on the request or load balance strategy
    model = current_model  # Default to current model
    if 'model_version' in data and data['model_version'] == 'new' and new_model is not None:
        model = new_model

    # Make predictions
    predictions = model.predict(df, num_iteration=model.best_iteration)
    return jsonify(predictions.tolist())

@app.route('/retrain', methods=['POST'])
def retrain():
    global new_model, LAST_RUN_ID
    file = request.files['file']
    df = pd.read_csv(file)

    target = "co2"
    X = df.drop(columns=[target])
    y = df[target]

    run_id, mse, new_model = train_and_log_model(X, y)
    LAST_RUN_ID = run_id
    new_model_path = os.path.join(MODEL_DIR, "new_model.pkl")
    joblib.dump(new_model, new_model_path)

    return jsonify({"status": "New model trained", "MSE": mse})

@app.route('/switch', methods=['POST'])
def switch():
    global current_model, new_model, CURRENT_RUN_ID, LAST_RUN_ID
    if new_model is not None:
        CURRENT_RUN_ID = LAST_RUN_ID
        current_model = new_model
        current_model_path = os.path.join(MODEL_DIR, "current_model.pkl")
        joblib.dump(current_model, current_model_path)
        new_model = None
        new_model_path = os.path.join(MODEL_DIR, "new_model.pkl")
        if os.path.exists(new_model_path):
            os.remove(new_model_path)
        return jsonify({"status": "Switched to new model"})
    else:
        return jsonify({"status": "No new model to switch to"})

@app.route('/statistics', methods=['GET'])
def statistics():
    global CURRENT_RUN_ID
    run = get_run(CURRENT_RUN_ID)
    stats = {
        "run_id": CURRENT_RUN_ID,
        "MSE": run.data.metrics["MSE"],
        "MAE": run.data.metrics["MAE"],
        "R2": run.data.metrics["R2"],
        "start_time": run.info.start_time
    }
    return jsonify(stats)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001)
