import pandas as pd
import mlflow
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestRegressor

from typing import Tuple

def train_model(model:RandomForestRegressor, run_name:str, x:pd.DataFrame, y:pd.DataFrame) -> str:
	signature = infer_signature(x,y)
	with mlflow.start_run(run_name=run_name) as run:
		model.fit(x, y)
		mlflow.sklearn.log_model(sk_model=model, artifact_path="model", signature=signature)
	return run.info.run_id