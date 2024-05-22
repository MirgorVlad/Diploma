import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load libraries for multiple models
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

def load_data() -> pd.DataFrame:
	# df = pd.read_csv("/home/thingsboard563/Private/ml_project/sensor_data.csv", sep=";")
	df = pd.read_csv("data/sensor_data.csv", sep=",")
	return df
