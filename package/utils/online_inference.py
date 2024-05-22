import pandas as pd
import json
import requests

def send_request(x: pd.DataFrame):
    features = [f for f in x.columns if f not in ['Timestamp', 'co2', 'noise', 'dust']]
    feature_values = json.loads(x[features].to_json(orient="split"))
    payload = {"dataframe_split": feature_values}

    BASE_URI = "http://127.0.0.1:5000/"
    headers = {"Content-Type": "application/json"}
    endpoint = BASE_URI + "invocations"
    r = requests.post(endpoint, data=json.dumps(payload), headers=headers)

    print(f"PREDICTIONS: {r.text}")
