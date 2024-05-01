from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split


def get_train_test_set(df: pd.DataFrame) -> Tuple:
	# Feature selection
	X = df.drop(columns=['Timestamp', 'co2', 'noise', 'dust'])

	# Target variable
	y = df['co2']

	# Split the data into training and testing sets
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
	return X_train, X_test, y_train, y_test