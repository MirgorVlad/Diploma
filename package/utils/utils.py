import mlflow
import pandas as pd
from typing import Dict
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

import matplotlib.pyplot as plt


def set_or_create_experiment(experiment_name: str) -> str:
    """
    Get or create an experiment.

    :param experiment_name: Name of the experiment.
    :return: Experiment ID.
    """

    try:
        experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
    except:
        experiment_id = mlflow.create_experiment(experiment_name)
    finally:
        mlflow.set_experiment(experiment_name=experiment_name)

    return experiment_id


def get_performance_plots_regr(
        y_true: pd.DataFrame, y_pred: pd.DataFrame
) -> Dict[str, any]:
    """
    Get performance plots for regression models.

    :param y_true: True labels.
    :param y_pred: Predicted labels.
    :param prefix: Prefix for the plot names.
    :return: Performance plots.
    """
    # Calculate regression metrics
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # Plot true vs predicted values
    plt.figure(figsize=(10, 5))
    plt.scatter(y_true, y_pred, color='blue', alpha=0.5)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color='red', linestyle='--')
    plt.title('True vs Predicted Values')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.grid(True)
    true_pred_figure = plt.gcf()
    plt.close()  # Close the figure to release memory

    # Plot residual plot
    plt.figure(figsize=(10, 5))
    residuals = y_true - y_pred
    plt.scatter(y_pred, residuals, color='blue', alpha=0.5)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.title('Residual Plot')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.grid(True)
    residual_figure = plt.gcf()
    plt.close()  # Close the figure to release memory

    return {
        "true_vs_pred": true_pred_figure,
        "residual_plot": residual_figure,
        "mse": mse,
        "mae": mae,
        "r2": r2
    }