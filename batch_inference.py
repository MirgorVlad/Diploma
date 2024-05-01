from package.feature.data_processing import preprocess_data
from package.ml_training.retrieal import get_train_test_score_set
from package.utils.utils import regression_report
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd

import mlflow

if __name__=="__main__":
    df = preprocess_data()
    x_train, x_test, x_score, y_train, y_test, y_score = get_train_test_score_set(df)
    features = [f for f in x_train.columns if f not in ["id", "target", "c02"]]

    model_uri = "models:/randomForestModel/latest"
    mlflow_model = mlflow.sklearn.load_model(model_uri=model_uri)

    predictions = mlflow_model.predict(x_score[features])

    scored_data = pd.DataFrame({"prediction": predictions, "target": y_score})
    report = regression_report(y_score, predictions)

    print(report)
    print(scored_data.head(10))


