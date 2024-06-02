import lightgbm as lgb
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn

from package.feature.data_processing import load_data
from package.utils.utils import set_or_create_experiment, get_performance_plots_regr

def initial_training(experiment_name="experiment"):
    df = load_data()
    target = "co2"

    X = df.drop(columns=[target])
    y = df[target]
    experiment_id = set_or_create_experiment(experiment_name)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create the LightGBM dataset
    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test)

    # Define parameters
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9
    }

    model = lgb.train(params, train_data, valid_sets=[test_data])
    y_pred = model.predict(X_test, num_iteration=model.best_iteration)

    performance_results = get_performance_plots_regr(y_test, y_pred)
    mse = performance_results["mse"]
    with mlflow.start_run(run_name="run") as run:
        mlflow.sklearn.log_model(model, "model")
        mlflow.log_params(params)
        mlflow.log_metric("MSE", mse)
        mlflow.log_metric("MAE", performance_results["mae"])
        mlflow.log_metric("R2", performance_results["r2"])
        mlflow.log_figure(performance_results["true_vs_pred"], "true_vs_pred.png")
        mlflow.log_figure(performance_results["residual_plot"], "residual_plot.png")

        return model

def train_and_log_model(X, y, experiment_name="experiment"):
    experiment_id = set_or_create_experiment(experiment_name)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create the LightGBM dataset
    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test)

    # Define parameters
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9
    }

    model = lgb.train(params, train_data, valid_sets=[test_data])
    y_pred = model.predict(X_test, num_iteration=model.best_iteration)


    performance_results = get_performance_plots_regr(y_test, y_pred)
    mse = performance_results["mse"]
    with mlflow.start_run(run_name="run") as run:
        mlflow.sklearn.log_model(model, "model")
        mlflow.log_params(params)
        mlflow.log_metric("MSE", mse)
        mlflow.log_metric("MAE", performance_results["mae"])
        mlflow.log_metric("R2", performance_results["r2"])
        mlflow.log_figure(performance_results["true_vs_pred"], "true_vs_pred.png")
        mlflow.log_figure(performance_results["residual_plot"], "residual_plot.png")

        return mlflow.active_run().info.run_id, mse, model