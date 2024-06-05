import pandas as pd


def load_data(path) -> pd.DataFrame:
	# df = pd.read_csv("/home/thingsboard563/Private/ml_project/sensor_data.csv", sep=";")
	df = pd.read_csv(path, sep=",")
	return df
