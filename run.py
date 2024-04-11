from package.feature.data_processing import preprocess_data
from package.utils.utils import set_or_create_experiment
from package.utils.utils import get_performance_plots_regr
from package.ml_training.train import train_model
from package.ml_training.retrieal import get_train_test_set
from sklearn.ensemble import RandomForestRegressor
import mlflow

if __name__ == '__main__':
	experiment_name = "model1"
	run_name = "test_1"

	df = preprocess_data()
	X_train, X_test, y_train, y_test = get_train_test_set(df)
	experiment_id = set_or_create_experiment(experiment_name=experiment_name)

	model = RandomForestRegressor(n_estimators=100, random_state=42)
	run_id = train_model(model, run_name, X_train, y_train)
	
	y_pred = model.predict(X_test)

	prefix = "test"
	performance_results = get_performance_plots_regr(y_test, y_pred, prefix)

# log performance metrics
with mlflow.start_run(run_id=run_id):
    mlflow.log_metric("MSE", performance_results["mse"])
    mlflow.log_metric("MAE", performance_results["mae"])
    mlflow.log_metric("R2", performance_results["r2"])

    # Log figures with MLflow
    mlflow.log_figure(performance_results[f"{prefix}_true_vs_pred"], prefix + "_true_vs_pred.png")
    mlflow.log_figure(performance_results[f"{prefix}_residual_plot"], prefix + "_residual_plot.png")