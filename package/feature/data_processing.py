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
	df = pd.read_csv("data/sensor_data.csv", sep=";")
	return df

def preprocess_data(df:pd.DataFrame) -> pd.DataFrame:
	# Parse timestamp column
	df['Timestamp'] = pd.to_datetime(df['Timestamp'])


	# Feature Engineering - extract temporal features
	df['hour'] = df['Timestamp'].dt.hour
	df['day_of_week'] = df['Timestamp'].dt.dayofweek
	df['month'] = df['Timestamp'].dt.month


	# Handle missing values (replace with mean)
	pd.set_option('future.no_silent_downcasting', True)
	df.fillna(df.mean(), inplace=True)
	df = df.infer_objects(copy=False)

	# # Feature Scaling
	# # Extract numerical columns (excluding 'Timestamp')
	# numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
	# # Scale numerical features
	# scaler = StandardScaler()
	# df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

	return df