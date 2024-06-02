import pandas as pd


def load_data() -> pd.DataFrame:
	# df = pd.read_csv("/home/thingsboard563/Private/ml_project/sensor_data.csv", sep=";")
	df = pd.read_csv("data/sensor_data.csv", sep=",")
	return df
