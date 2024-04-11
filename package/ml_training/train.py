import pandas as pd
import mlflow
from sklearn.ensemble import RandomForestRegressor

from typing import Tuple

def train_model(model:RandomForestRegressor, run_name:str, x:pd.DataFrame, y:pd.DataFrame) -> str:
	with mlflow.start_run(run_name=run_name) as run:
		model.fit(x, y)
		mlflow.sklearn.log_model(model, "model")
	return run.info.run_id