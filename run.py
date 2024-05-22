import mlflow
from package.feature.data_processing import *
from package.utils.utils import set_or_create_experiment
from package.utils.utils import get_performance_plots_regr
from package.ml_training.train import train_model
from package.ml_training.retrieal import get_train_test_set
from sklearn.ensemble import RandomForestRegressor
from package.utils.data_generator import generate_random_data_with_trends
from package.utils.online_inference import send_request

if __name__ == '__main__':
    model_name = "rfm"
    experiment_name = "co2 prediction"
    run_name = "run01"
    artifact_path = "artifacts"

    df = preprocess_data(load_data())
    X_train, X_test, y_train, y_test = get_train_test_set(df)
    experiment_id = set_or_create_experiment(experiment_name=experiment_name)

    n_estimators = 100
    random_state = 42

    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    run_id = train_model(model, run_name, X_train, y_train)

    y_pred = model.predict(X_test)

    prefix = "test"
    performance_results = get_performance_plots_regr(y_test, y_pred, prefix)

    mlflow.register_model(model_uri=f"runs:/{run_id}/{artifact_path}", name=model_name)
    # Log performance metrics
    with mlflow.start_run(run_id=run_id):
        # Log parameters
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("random_state", random_state)

        # Set tags
        mlflow.set_tag("experiment_name", experiment_name)
        mlflow.set_tag("run_name", run_name)

        # Log performance metrics
        mlflow.log_metric("MSE", performance_results["mse"])
        mlflow.log_metric("MAE", performance_results["mae"])
        mlflow.log_metric("R2", performance_results["r2"])

        # Log figures with MLflow
        mlflow.log_figure(performance_results[f"{prefix}_true_vs_pred"], prefix + "_true_vs_pred.png")
        mlflow.log_figure(performance_results[f"{prefix}_residual_plot"], prefix + "_residual_plot.png")
