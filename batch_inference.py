from package.feature.data_processing import preprocess_data
from package.ml_training.retrieal import get_train_test_score_set
from package.utils.utils import regression_report
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd

import mlflow

if __name__=="__main__":
    model_uri = "models:/randomForestModel/latest"
    mlflow_model = mlflow.sklearn.load_model(model_uri=model_uri)




